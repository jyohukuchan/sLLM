# AQ low-latency dequant progress

## 前回の要点

- 現行AQ4 prototypeは、controlled v0.3 prompt suiteでR9700/RDNA4 mean decode `19.796 tok/s`、V620/RDNA2 mean decode `15.434 tok/s` に到達している。
- 次の本命は、AQをf32に全展開せず、compact payloadとscaleを低遅延に読みながら推論へ接続すること。
- token/sだけでなく、v0.3 prompt suite、golden prefix、top-logits guardで出力品質の崩壊を同時に見る必要がある。

## 今回の変更点

- `tools/build-sq-candidate-runtime-row.py` で、prompt-suite summaryとguard bundleから `sq-candidate-runtime-result-v0.1` JSONL行を生成できる状態を確認した。
- 現行AQ4 baselineとして、R9700/RDNA4とV620/RDNA2の2行を `benchmarks/results/2026-07-06/engine/sq-candidate-runtime-baseline-aq4-v0.1.jsonl` に生成した。
- baseline anchor rowでは、現行AQ4 runtimeで未計測のcompact resident bytes、materialized working-set bytes、materialization wall timeを `null` にできる例外を仕様へ明記した。
- ビルダーの単体テストを追加し、baseline anchor rowと通常候補rowのgate判定を固定した。

## 次の行動

1. 既存runtimeのAQ4 materialize kernel、matvec kernel、prompt-suite runnerを読んで、f32全展開なしのdequant+matvec APIを追加する最小境界を決める。
2. まず単一projectionまたはlm-headを対象に、compact AQ4 payloadを直接読み、raw codebook値とscaleを分けて累積するベンチを作る。
3. R9700でwarm-upを明示した短いbenchを回し、同じartifactでguard bundleまたはv0.3 suiteの一部を通して、token/sと出力品質が同時に壊れないかを見る。

## 2026-07-06 resident self-attn追加

### 前回の要点

- AQ4 direct matvec自体は既にcompact idx/scale/codebookを直接読み、f32全展開なしで動いていた。
- 旧R9700 v0.3 suiteはmean decode `19.796 tok/s` で、同サイズllama.cpp Q4_K_XLの `~79 tok/s` には届いていなかった。
- 旧decode内訳ではself-attn層が約 `40ms/token` を占め、linear-attn層はdevice-resident連鎖済みだった。

### 今回の変更点

- Qwen3.5 gated q projectionをdevice上で `query` と `output gate` に分ける `qwen35_split_q_gate_f32` runtime ABI/HIPRTC kernelを追加した。
- `package-token-ids-bench` のself-attn層を、q/k/v/oとMLP gate/up/downをAQ4 resident matvecで直接実行する経路へ置き換えた。
- decode layer loopをresident layer全般でdevice-to-device連鎖できる形にし、self-attn境界でのhost read/writeとf32 materialized self-attn weightを外した。
- R9700 prompt16/gen128でdecode `23.592 tok/s`、skip2 `23.593 tok/s`、last8 `23.249 tok/s`、prefill `18.439 tok/s` を得た。
- v0.3 prompt suite相当を新経路で実行し、mean decode `23.154 tok/s`、min `21.543 tok/s`、max `23.707 tok/s`、mean prefill `22.273 tok/s`、verified all `true` を得た。
- output healthは `ok=1`、`warn=5`、`not_evaluated=1`。速度改善で壊滅はしていないが、control marker、prompt echo、反復、generation limitの警告は残った。
- 部分実行ではself-attn 8層のみが `60.767 tok/s`、linear-attn 24層のみが `29.696 tok/s`。full 32層のlayers平均は `33.868ms/token`、lm_head平均は `7.137ms/token`。

### 次の行動

1. AQ4フォーマット自体は今回の壁ではない。次の主因はlinear-attn層の多数kernel launch/逐次処理と、gpu_resident_f32 lm_headの約 `7ms/token`。
2. 既存量子化の `~79 tok/s` と比較するなら、linear-attn stepのkernel融合、GPU-side top-kまたは量子化lm_head、prefillのhost readback削減を優先する。
3. 出力品質は速度変更による崩壊ではなく、既存のプロンプト/停止条件/生成制御問題として別途guardを強める。

## 2026-07-06 top1/final-norm/BF16 lm_head追加

### 前回の要点

- resident self-attn後のR9700 prompt16/gen128はdecode `23.592 tok/s`、lm_head平均 `7.137ms/token` だった。
- v0.3 prompt suite相当ではmean decode `23.154 tok/s`、verified all `true` まで到達していた。
- 次の候補はlm_head host readback、final RMSNormのCPU境界、BF16 passthroughのf32 resident展開だった。

### 今回の変更点

- GPU-side top1 partial reductionを追加し、top_k=1ではvocab全logitsをhostへ戻さず、970個のpartialだけをreadbackするようにした。
- decode時のfinal RMSNormをGPU resident化し、最終層output bufferからlm_headへdevice上で接続した。
- lm_head BF16 passthroughをf32へ全展開せず、BF16 raw payloadをチャンクコピーでGPUへ置き、`matvec_bf16_f32` kernelでf32入力と積和する経路を追加した。
- R9700 prompt16/gen128:
  - top1のみ: decode `23.835 tok/s`、lm_head平均 `6.605ms/token`。
  - top1 + GPU final RMSNorm: decode `24.030 tok/s`、出力列はbaselineと一致。
  - top1 + GPU final RMSNorm + BF16 lm_head: decode `24.466 tok/s`、last8 `24.275 tok/s`、verified `true`、出力列はbaselineと一致。
- BF16 lm_headでresident matrix bytesは `4,068,474,880` から `2,034,237,440` に減り、lm_head loadは約 `5.2s` から約 `0.205s` になった。
- V620/RDNA2 prompt16/gen32でもBF16 lm_head版が完走し、decode `20.086 tok/s`、last8 `20.191 tok/s`、verified `true` を得た。
- R9700 v0.3 prompt suite相当をtop_k=1で実行し、mean decode `24.185 tok/s`、min `22.487 tok/s`、max `24.706 tok/s`、mean prefill `21.755 tok/s`、verified all `true` を得た。
- output healthは `ok=1`、`warn=5`、`not_evaluated=1`。警告の種類は既存resident self-attn版と同じで、速度変更による壊滅的な品質崩壊は観測していない。

### 次の行動

1. R9700/RDNA4とV620/RDNA2の両方で20 tok/s以上を確認できたので、一部GPUで動くprototypeの速度目標は一旦満たしたと考える。
2. AQ format自体の壁にはまだ当たっていない。次の主因は、32層内のAQ4 matvec群とlinear-attn stepの逐次kernel構成で、lm_head単独最適化の余地は小さくなった。
3. 既存量子化engine級の `~79 tok/s` を狙うなら、次はlinear-attn recurrent stepのkernel融合、AQ4 matvecの複数projection融合、embedding resident化またはtoken embedding cacheを検討する。

## 2026-07-06 BF16 embedding resident追加

### 前回の要点

- top1、GPU final RMSNorm、BF16 lm_head追加後のR9700 prompt16/gen128はdecode `24.466 tok/s`、v0.3 prompt suite相当はmean decode `24.185 tok/s` だった。
- V620/RDNA2 prompt16/gen32でもdecode `20.086 tok/s` に到達しており、一部GPUで動くprototypeとしての最低速度目標は満たし始めていた。
- 残っていた小さいhost境界として、decodeごとのembedding row読み出しとhost-to-device入力コピーがあった。

### 今回の変更点

- BF16 passthrough embeddingをraw BF16のままGPUへ常駐させ、選択token rowだけをf32 residual bufferへ変換する `bf16_row_f32` runtime ABI/HIPRTC kernelを追加した。
- `package-token-ids-generate` のdecode経路で、embedding row読み出しからlayer stack入力までをdevice上で接続した。
- 初回HIPRTC compileがdecode平均に混ざるとtoken/sが低く見えるため、embedding runtime load時にrow gather kernelをprewarmするようにした。非prewarm版は参考値扱いにする。
- R9700 prompt16/gen128ではdecode `25.300 tok/s`、skip2 `25.297 tok/s`、last8 `24.813 tok/s`、prefill `18.370 tok/s` を得た。embedding gather平均は `0.007ms/token`、最大 `0.041ms/token` だった。
- R9700の生成token列は、BF16 lm_headのみの前段artifactと一致し、verified `true` だった。
- V620/RDNA2 prompt16/gen32ではdecode `20.910 tok/s`、skip2 `21.098 tok/s`、last8 `21.029 tok/s`、prefill `14.953 tok/s` を得た。生成token列は前段artifactと一致し、verified `true` だった。
- R9700 v0.3 prompt suite相当ではmean decode `24.766 tok/s`、min `22.851 tok/s`、max `25.408 tok/s`、mean prefill `22.043 tok/s`、verified all `true` を得た。
- output healthは `ok=1`、`warn=5`、`not_evaluated=1`。速度変更による壊滅的な品質崩壊は観測していない。

### 次の行動

1. embedding host境界は実質的に解消した。ここでAQ format自体の壁にはまだ直面していない。
2. 単発benchではR9700 `25.30 tok/s`、V620 `20.91 tok/s` まで上がったが、既存量子化engineの `~79 tok/s` とはまだ大きな差がある。
3. 次の主因は、層内AQ4 matvec群とlinear-attn recurrent stepのkernel launch/同期構成なので、複数projection融合とlinear-attn step融合を優先する。

## 2026-07-06 AQ4 matvec residual add融合

### 前回の要点

- BF16 embedding resident後のR9700 prompt16/gen128はdecode `25.300 tok/s`、V620 prompt16/gen32はdecode `20.910 tok/s` だった。
- R9700 v0.3 prompt suite相当はmean decode `24.766 tok/s`、verified all `true` だった。
- まだ既存量子化engine級の `~79 tok/s` には遠く、次の主因は層内AQ4 matvec群とlinear-attn recurrent stepのkernel launch/同期構成だった。

### 今回の変更点

- `aq4_matvec_add_f32` runtime ABI/HIPRTC kernelを追加し、AQ4 matvecの出力にresidual bufferを同一kernel内で加算できるようにした。
- self-attn層とlinear-attn層の `o/out projection -> residual add` と `MLP down -> residual add` を fused AQ4 matvec addに置き換えた。
- 初回HIPRTC compileがprefillに混ざらないよう、resident layer load時にAQ4 matvec add kernelをprewarmするようにした。
- R9700 prompt16/gen128ではdecode `25.445 tok/s`、skip2 `25.444 tok/s`、last8 `24.967 tok/s`、prefill `17.710 tok/s` を得た。BF16 embedding resident版比で `+0.145 tok/s`、約 `+0.57%`。
- V620/RDNA2 prompt16/gen32ではdecode `20.999 tok/s`、skip2 `21.150 tok/s`、last8 `21.070 tok/s`、prefill `16.137 tok/s` を得た。BF16 embedding resident版比で `+0.090 tok/s`、約 `+0.43%`。
- R9700とV620の生成token列は、それぞれBF16 embedding resident版の前段artifactと一致し、verified `true` だった。
- R9700 v0.3 prompt suite相当ではmean decode `24.922 tok/s`、min `22.977 tok/s`、max `25.544 tok/s`、mean prefill `22.150 tok/s`、verified all `true` を得た。BF16 embedding resident suite比で `+0.156 tok/s`、約 `+0.63%`。
- output healthは `ok=1`、`warn=5`、`not_evaluated=1`。速度変更による壊滅的な品質崩壊は観測していない。

### 次の行動

1. residual add融合は正方向だが効果は1%未満なので、AQ4フォーマット自体の壁とは言えない。
2. 既存量子化engine級の `~79 tok/s` との差はまだ大きい。次は単独addのような小kernel削減ではなく、linear-attn recurrent/qkv prepare/post処理の融合か、qkv/z/a/bなど複数AQ4 projectionの同時処理が必要。
3. 特にlinear-attn層は24層あるため、linear-attn stepのkernel数削減を優先する。

## 2026-07-06 linear-attn qkv prepare 1-kernel化

### 前回の要点

- AQ4 matvec residual add融合後のR9700 prompt16/gen128はdecode `25.445 tok/s`、V620 prompt16/gen32はdecode `20.999 tok/s` だった。
- R9700 v0.3 prompt suite相当はmean decode `24.922 tok/s`、verified all `true` だった。
- 次の候補は、24層あるlinear-attn層の細かいkernel launch削減だった。

### 今回の変更点

- `linear_attn_qkv_prepare_f32` のHIP pathを、従来の `conv+silu` kernel と `split+l2norm` kernel の2段構成から、1つのHIPRTC kernelへ統合した。
- q/k headはconv履歴更新、SiLU、head内L2 norm、q scale適用までを同一blockで行い、vはconv履歴更新、SiLU、v出力コピーまでを同一kernelで処理する。
- 既存ABIは維持し、runtime内部のHIP kernel cacheとlaunch経路だけを差し替えた。
- 初回HIPRTC compileがprefill測定へ混ざらないよう、resident layer load時にHIPデバイス単位でqkv prepare kernelをprewarmするようにした。prewarm後はconv historyをゼロへ戻し、推論開始状態を変えない。
- R9700 prompt16/gen128ではdecode `25.485 tok/s`、skip2 `25.484 tok/s`、last8 `24.988 tok/s`、prefill `19.522 tok/s` を得た。matvecadd版比でdecode `+0.040 tok/s`、約 `+0.16%`、prefill `+1.811 tok/s`。
- V620/RDNA2 prompt16/gen32ではdecode `21.020 tok/s`、skip2 `21.194 tok/s`、last8 `21.120 tok/s`、prefill `15.883 tok/s` を得た。matvecadd版比でdecode `+0.021 tok/s`、約 `+0.10%`。
- R9700とV620の生成token列は、それぞれmatvecadd版の前段artifactと一致し、verified `true` だった。
- R9700 v0.3 prompt suite相当ではmean decode `25.008 tok/s`、min `23.062 tok/s`、max `25.639 tok/s`、mean prefill `22.986 tok/s`、verified all `true` を得た。matvecadd suite比でmean decode `+0.086 tok/s`、約 `+0.35%`、mean prefill `+0.836 tok/s`。
- output healthは `ok=1`、`warn=5`、`not_evaluated=1` のまま変化なし。速度変更による壊滅的な品質崩壊は観測していない。

### 次の行動

1. qkv prepare 2-kernel削減のdecode効果は小さいため、ここもAQ4フォーマット自体の壁ではない。
2. 現在のR9700 full decodeは約 `25 tok/s` で、君が目安にしていた `15-20 tok/s` は超えた。一方、既存量子化engine級の `~79 tok/s` とはまだ差が大きい。
3. 次に速度を伸ばすなら、linear-attn postの `segmented_rmsnorm + silu_mul` 融合、linear-attn recurrent stepの内部融合、または複数AQ4 projectionの同時処理に進む。特にlm_head/step wallの支配が強いため、単発の小kernel融合だけでは伸びにくい。

## 2026-07-06 linear-attn post RMSNorm/SiLU-mul融合

### 前回の要点

- linear-attn qkv prepare 1-kernel化後のR9700 prompt16/gen128はdecode `25.485 tok/s`、V620 prompt16/gen32はdecode `21.020 tok/s` だった。
- R9700 v0.3 prompt suite相当はmean decode `25.008 tok/s`、verified all `true` だった。
- qkv prepareの効果は小さく、次の小kernel削減候補はlinear-attn postの `segmented_rmsnorm + silu_mul` だった。

### 今回の変更点

- `segmented_rmsnorm_silu_mul_f32` runtime ABI/HIPRTC kernelを追加し、segmentごとのRMSNorm直後に `silu(z) * normed` を同一kernelで出力できるようにした。
- resident linear-attn stepでは中間 `attn_normed_buffer` を外し、`recurrent_output + attn_norm_weight + z` から直接 `attn_projection_input` を作る経路へ置き換えた。
- 初回HIPRTC compileが測定へ混ざらないよう、resident layer load時にHIPデバイス単位でpost fused kernelをprewarmするようにした。
- R9700 prompt16/gen128ではdecode `25.557 tok/s`、skip2 `25.555 tok/s`、last8 `25.089 tok/s`、prefill `18.674 tok/s` を得た。qkvprepare版比でdecode `+0.072 tok/s`、約 `+0.28%`。
- V620/RDNA2 prompt16/gen32ではdecode `21.075 tok/s`、skip2 `21.228 tok/s`、last8 `21.153 tok/s`、prefill `16.923 tok/s` を得た。qkvprepare版比でdecode `+0.055 tok/s`、約 `+0.26%`。
- R9700とV620の生成token列は、それぞれqkvprepare版の前段artifactと一致し、verified `true` だった。
- R9700 v0.3 prompt suite相当ではmean decode `25.019 tok/s`、min `23.065 tok/s`、max `25.656 tok/s`、mean prefill `22.822 tok/s`、verified all `true` を得た。qkvprepare suite比でmean decode `+0.0105 tok/s`、約 `+0.04%`。
- output healthは `ok=1`、`warn=5`、`not_evaluated=1` のまま変化なし。速度変更による壊滅的な品質崩壊は観測していない。

### 次の行動

1. post融合は安全だが、suite平均ではほぼノイズ幅なので、AQ4フォーマット自体の壁とは言えない。
2. 単発・suiteともR9700は約 `25 tok/s`、V620は約 `21 tok/s` に到達しており、RDNA4/RDNA2のプロトタイプとして「動く」水準は満たしている。
3. 既存量子化engine級の差を詰めるなら、次は小kernel削減ではなく、LM head BF16 matvecの実効計算性能、AQ4複数projection同時処理、またはlinear-attn recurrent前後の大きめ融合を見るべき。

## 2026-07-06 BF16 lm_head 64-thread row kernel

### 前回の要点

- linear-attn post RMSNorm/SiLU-mul融合後のR9700 prompt16/gen128はdecode `25.557 tok/s`、V620 prompt16/gen32はdecode `21.075 tok/s` だった。
- R9700 v0.3 prompt suite相当はmean decode `25.019 tok/s`、verified all `true` だった。
- decode step内訳ではR9700のlm_head平均が `38.104ms/token`、step wall平均が `39.128ms/token` で、次の支配要因はAQ4 dequantではなくBF16 lm_head matvecだった。

### 今回の変更点

- BF16 lm_head用の `matvec_bf16_f32_hip_kernel` launchを、1行あたり `256` threadsからAMD wavefront相当の `64` threadsへ変更した。
- R9700 prompt16/gen128ではdecode `26.670 tok/s`、skip2 `26.666 tok/s`、last8 `26.122 tok/s`、prefill `19.023 tok/s` を得た。postfused版比でdecode `+1.113 tok/s`、約 `+4.35%`。
- R9700のlm_head平均は `38.104ms/token` から `36.585ms/token` へ、step wall平均は `39.128ms/token` から `37.495ms/token` へ下がった。
- V620/RDNA2 prompt16/gen32ではdecode `21.084 tok/s`、skip2 `21.265 tok/s`、last8 `21.198 tok/s`、prefill `17.028 tok/s` を得た。postfused版比でdecode `+0.008 tok/s` とほぼ中立だった。
- R9700とV620の生成token列は、それぞれpostfused版の前段artifactと一致し、verified `true` だった。
- R9700 v0.3 prompt suite相当ではmean decode `26.162 tok/s`、min `23.967 tok/s`、max `26.851 tok/s`、mean prefill `22.914 tok/s`、verified all `true` を得た。postfused suite比でmean decode `+1.143 tok/s`、約 `+4.57%`。
- suiteのoutput healthは `ok=1`、`warn=5`、`not_evaluated=1` のまま変化なし。速度変更による壊滅的な品質崩壊は観測していない。

### 次の行動

1. 今回の改善はAQ4フォーマット自体の壁ではなく、BF16 lm_head matvec launch構成の問題だった。
2. R9700は単発 `26.67 tok/s`、suite平均 `26.16 tok/s` まで上がり、君が当初見込んでいたFP32換算の最低線 `15-20 tok/s` は明確に超えている。一方、既存量子化engine級の `~79 tok/s` との差はまだ大きい。
3. 次はlm_headのさらなる低遅延化、特にmatvecとtop1の統合、またはAQ4複数projection同時処理を検討する。小kernel融合だけでは大きな伸びは出にくくなっている。

## 2026-07-06 resident BF16 lm_head prewarm

### 前回の要点

- BF16 lm_head 64-thread row kernel後のR9700 prompt16/gen128はdecode `26.670 tok/s`、V620 prompt16/gen32はdecode `21.084 tok/s` だった。
- R9700 v0.3 prompt suite相当はmean decode `26.162 tok/s`、mean prefill `22.914 tok/s`、verified all `true` だった。
- prefillのlm_head部分に初回HIPRTC compileが混ざり、prefill token/sを低く見せる可能性が残っていた。

### 今回の変更点

- `PackageLmHeadRuntime::load` のGPU resident BF16/F32 lm_head経路で、ゼロ入力を使ったmatvec + top1を1回実行し、初回HIPRTC compileとtop1 partial readback経路をload時にprewarmするようにした。
- R9700 prompt16/gen128では生成token列がlmhead64版と一致し、verified `true` だった。decodeは `26.671 tok/s`、skip2 `26.661 tok/s`、last8 `26.143 tok/s` で、lmhead64版と実質同等だった。
- R9700のprefillは `19.023 tok/s` から `19.675 tok/s` へ上がり、prefill内のlm_head wallは `37.628ms` から `3.382ms` へ下がった。lm_head loadは `200.533ms` から `239.977ms` へ増えた。
- V620/RDNA2 prompt16/gen32でも生成token列がlmhead64版と一致し、verified `true` だった。decodeは `21.274 tok/s`、skip2 `21.268 tok/s`、last8 `21.192 tok/s` で、skip2/last8は実質同等だった。
- V620のprefillは `17.028 tok/s` から `17.542 tok/s` へ上がり、prefill内のlm_head wallは `44.001ms` から `6.091ms` へ下がった。lm_head loadは `185.852ms` から `227.905ms` へ増えた。
- R9700 v0.3 prompt suite相当ではmean decode `26.128 tok/s`、min `23.978 tok/s`、max `26.814 tok/s`、mean prefill `23.709 tok/s`、verified all `true` を得た。lmhead64 suite比でmean decodeは `-0.034 tok/s` と実質同等、mean prefillは `+0.794 tok/s`、約 `+3.47%`。
- suiteのprefill lm_head wall平均は `41.020ms` から `3.388ms` へ下がり、lm_head load平均は `190.668ms` から `225.271ms` へ増えた。
- suiteのoutput healthは `ok=1`、`warn=5`、`not_evaluated=1` のまま変化なし。速度・測定位置の変更による壊滅的な品質崩壊は観測していない。

### 次の行動

1. prewarmはdecode高速化ではなく、初回compileをprefillからloadへ移す測定品質改善として採用する。君が指摘していたwarmup問題には正式に対応できた。
2. decodeの `lm_head_step_ms` は、直前のlayer kernelが非同期実行され、top1 partial readback時の同期にまとめて乗っている可能性が高い。prefill prewarm後のlm_head wallが数msまで下がったため、decode内訳を「純粋なlm_head時間」と解釈しない。
3. 次に既存量子化engine級との差を詰めるなら、lm_head単体よりも、decode step全体のGPU queueを減らす方向、つまりAQ4複数projection同時処理、linear-attn recurrent前後の大きめ融合、または正確なGPU event計測によるボトルネック再分解を優先する。

## 2026-07-06 AQ4 matvec RDNA4 row block tuning

### 前回の要点

- resident BF16 lm_head prewarm後のR9700 prompt16/gen128はdecode `26.671 tok/s`、V620 prompt16/gen32はdecode `21.274 tok/s` だった。
- R9700 v0.3 prompt suite相当はmean decode `26.128 tok/s`、mean prefill `23.709 tok/s`、verified all `true` だった。
- prewarm後、prefillのlm_head wallは数msまで下がり、decodeの大きな `lm_head_step_ms` bucketは非同期layer stackの待ちを含むと判断した。

### 今回の変更点

- `ULLM_SYNC_DECODE_LAYERS_FOR_TIMING=1` 診断モードを追加し、通常benchには影響させずにdecode layer stack直後で同期できるようにした。
- R9700同期診断では、prewarm baselineのdecode step平均 `36.979ms` のうち、layer stack平均が `33.588ms`、lm_head平均が `3.388ms` だった。主因はlm_headではなくAQ4 layer stack側だった。
- AQ4 matvec/add/silu-mul/gate-betaのHIP launch block sizeを実験し、R9700/RDNA4では `256 -> 128 -> 64` threads/rowの順に改善した。同期診断gen32ではdecode `27.042 -> 30.431 -> 33.271 tok/s`、layer stack平均 `33.588ms -> 29.478ms -> 26.677ms` だった。
- V620/RDNA2では逆に `256` threads/rowが最速で、`128` はdecode `20.136 tok/s`、`64` はdecode `17.797 tok/s` まで悪化した。
- 正式実装は、HIP compute capability `major >= 12` のRDNA4系だけAQ4 matvec系を `64` threads/rowにし、それ以外は既存の `256` threads/rowを維持する方針にした。
- commit後のR9700 prompt16/gen128では生成token列がprewarm baselineと一致し、verified `true` だった。decode `32.660 tok/s`、skip2 `32.639 tok/s`、last8 `31.877 tok/s`、prefill `22.637 tok/s` を得た。prewarm baseline比でdecode `+5.988 tok/s`、約 `+22.45%`。
- commit後のV620/RDNA2 prompt16/gen32でも生成token列がprewarm baselineと一致し、verified `true` だった。decode `21.288 tok/s`、skip2 `21.281 tok/s`、last8 `21.205 tok/s` で、prewarm baseline比は実質中立だった。
- commit後のR9700同期診断gen32では、decode `33.284 tok/s`、layer stack平均 `26.656ms`、lm_head平均 `3.385ms`、step平均 `30.045ms` を得た。prewarm baseline同期診断比でlayer stack平均は `-6.933ms`、約 `-20.64%`。
- R9700 v0.3 prompt suite相当ではmean decode `32.013 tok/s`、min `29.071 tok/s`、max `32.840 tok/s`、mean prefill `28.465 tok/s`、verified all `true` を得た。prewarm suite比でmean decode `+5.886 tok/s`、約 `+22.53%`、mean prefill `+4.756 tok/s`、約 `+20.06%`。
- suiteのoutput healthは `ok=1`、`warn=5`、`not_evaluated=1` のまま変化なし。速度変更による壊滅的な品質崩壊は観測していない。

### 次の行動

1. AQ4フォーマット自体の壁ではなく、RDNA4上でのAQ4 matvec launch構成が大きな問題だった。1 wavefront/row相当がR9700では明確に効いた。
2. R9700は単発 `32.66 tok/s`、suite平均 `32.01 tok/s` まで上がったが、既存量子化engine級の `~79 tok/s` とはまだ差がある。
3. 次の候補はAQ4複数projection同時処理、特にlinear-attn層のq/k/v/z/a/b周辺のprojection融合。今回の結果から、層内AQ4 matvecのlaunchとrow並列が主要な改善対象だと見てよい。

## 2026-07-06 RDNA4 AQ4 multi-row wave grouping

### 前回の要点

- AQ4 matvec RDNA4 row block tuning後のR9700 prompt16/gen128はdecode `32.660 tok/s`、V620 prompt16/gen32はdecode `21.288 tok/s` だった。
- R9700 v0.3 prompt suite相当はmean decode `32.013 tok/s`、mean prefill `28.465 tok/s`、verified all `true` だった。
- R9700同期診断では、layer stack平均 `26.656ms`、lm_head平均 `3.385ms` で、まだlayer stackが支配的だった。

### 今回の変更点

- AQ4 matvec/add/silu-mul/gate-betaのHIP kernel内部に `rows_per_block` を追加し、RDNA4では `block=256, rows_per_block=4`、つまり4 wavefrontで4 rowを同一block内にまとめて処理するようにした。
- RDNA2など `major < 12` のGPUでは `rows_per_block=1` のままにして、既存の256 threads/row相当を維持した。
- R9700 prompt16/gen128では生成token列が前段AQ4 arch版と一致し、verified `true` だった。decode `34.420 tok/s`、skip2 `34.401 tok/s`、last8 `33.568 tok/s`、prefill `25.073 tok/s` を得た。前段比でdecode `+1.760 tok/s`、約 `+5.39%`。
- V620/RDNA2 prompt16/gen32でも生成token列は前段AQ4 arch版と一致し、verified `true` だった。decode `20.898 tok/s`、skip2 `20.891 tok/s`、last8 `20.823 tok/s` で、前段比では約 `-1.83%`。20 tok/s台は維持しているが、RDNA2ではmulti-row対応による小幅なオーバーヘッドが残った。
- R9700 v0.3 prompt suite相当ではmean decode `33.779 tok/s`、min `30.493 tok/s`、max `34.765 tok/s`、mean prefill `29.764 tok/s`、verified all `true` を得た。AQ4 arch suite比でmean decode `+1.765 tok/s`、約 `+5.51%`、mean prefill `+1.300 tok/s`、約 `+4.57%`。
- R9700 stack同期診断gen32では、decode `34.671 tok/s`、layer stack平均 `25.462ms`、lm_head平均 `3.378ms`、step平均 `28.842ms` を得た。前段比でlayer stack平均は `-1.193ms`、約 `-4.48%`。
- R9700層別同期診断gen32では、linear-attention 24層の平均が `0.828ms/layer`、合計 `19.879ms`、self-attention 8層の平均が `0.794ms/layer`、合計 `6.355ms` だった。前段より両方とも約4-5%短縮した。
- suiteのoutput healthは `ok=1`、`warn=5`、`not_evaluated=1` のまま変化なし。速度変更による壊滅的な品質崩壊は観測していない。

### 次の行動

1. RDNA4ではAQ4 matvecのrow処理をwave単位にし、さらにblock内で複数rowをまとめる方針が有効だった。AQ4フォーマット自体の壁にはまだ当たっていない。
2. R9700は単発 `34.42 tok/s`、suite平均 `33.78 tok/s` まで上がった。一方、V620は小幅に落ちたため、RDNA2での分岐オーバーヘッド削減か、RDNA2用に旧kernel pathを維持する検討余地がある。
3. 次に大きく伸ばすなら、単一matvecのrow groupingだけではなく、層内の複数AQ4 projectionを同一kernelまたは同一read pathで処理する方向へ進む。特に24層あるlinear-attention層のprojection群が合計時間の大部分を占める。

## 2026-07-06 AQ4 rows_per_block compile-time specialization

### 前回の要点

- RDNA4 AQ4 multi-row wave grouping後のR9700 prompt16/gen128はdecode `34.420 tok/s`、V620 prompt16/gen32はdecode `20.898 tok/s` だった。
- RDNA4では `block=256, rows_per_block=4` が有効だったが、RDNA2では `rows_per_block=1` のままでもruntime引数化による小さいオーバーヘッドが残っている可能性があった。
- 次の確認対象は、RDNA2での小幅回帰を消しつつ、RDNA4のmulti-row効果を維持できるかだった。

### 今回の変更点

- AQ4 matvec/add/silu-mul/gate-betaの `rows_per_block` をruntime kernel引数からHIPRTCソース生成時の `ULLM_AQ4_ROWS_PER_BLOCK` defineへ移した。
- `gfx12*` では `ULLM_AQ4_ROWS_PER_BLOCK=4`、それ以外では `1` をcompile-timeに埋め込み、RDNA2では動的な `blockDim.x / rows_per_block` と余計なkernel引数を避けられるようにした。
- code commitは `677237f Specialize AQ4 row grouping at compile time`。
- R9700 prompt16/gen128では生成token列がmulti-row版と一致し、verified `true` だった。decode `34.637 tok/s`、skip2 `34.616 tok/s`、last8 `33.827 tok/s`、prefill `25.070 tok/s` を得た。前段multi-row版比でdecode `+0.217 tok/s`、約 `+0.63%`。
- V620/RDNA2 prompt16/gen32でも生成token列がmulti-row版と一致し、verified `true` だった。decode `21.291 tok/s`、skip2 `21.284 tok/s`、last8 `21.212 tok/s`、prefill `16.643 tok/s` を得た。前段multi-row版比でdecode `+0.392 tok/s`、約 `+1.88%`。multi-row導入で出ていたRDNA2の小幅回帰はほぼ解消した。
- prompt suite wrapperのrequired HIP kernel listに `ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL` と `ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL` を追加し、prompt suite計測でstaging fallbackが混ざらないようにした。
- R9700 controlled v0.3 prompt suiteではmean decode `33.612 tok/s`、min `30.284 tok/s`、max `34.351 tok/s`、mean prefill `29.517 tok/s`、verified all `true` を得た。output healthは `ok=5`、`warn=1`、`not_evaluated=1`。warnは `throughput_checklist_direct` の `low_unique_token_ratio` で、速度変更による数値崩壊ではなく反復出力だった。
- V620/RDNA2 controlled v0.3 prompt suiteではmean decode `20.476 tok/s`、min `18.918 tok/s`、max `20.870 tok/s`、mean prefill `20.281 tok/s`、verified all `true` を得た。output healthはR9700と同じく `ok=5`、`warn=1`、`not_evaluated=1`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardでは、7/7 caseでprompt token、生成token、stop reason、verified、output statusが一致した。top logitsは `atol=1e-3` で7/7一致し、最大差はprefill `8.58e-6`、decode末尾 `2.99e-4` だった。

### 次の行動

1. RDNA2/RDNA4の一部GPUで動くAQ4プロトタイプとしては、controlled prompt suiteでR9700平均 `33.6 tok/s`、V620平均 `20.5 tok/s`、cross-device生成token一致まで確認できたため、発表可能な最低速度・品質崩壊チェックは満たしたと考える。
2. 一方、既存高速量子化engine級の `~79 tok/s` と比較するとまだ十分ではない。AQ4フォーマット自体の壁というより、24層あるlinear-attention側の複数AQ4 projectionとlayer stack全体のlaunch/read pathが次の主戦場。
3. これ以上伸ばすなら、単一AQ4 matvec kernel内の微調整ではなく、linear-attention層の複数projectionをまとめる設計、または層別同期診断で残り時間を再分解して、最も大きいprojection群から融合する。

## 2026-07-06 RDNA4 AQ4 rows16 grouping and AQ4 variant prewarm

### 前回の要点

- AQ4 rows_per_block compile-time specialization後のR9700 prompt16/gen128はdecode `34.637 tok/s`、V620 prompt16/gen32はdecode `21.291 tok/s` だった。
- R9700 controlled v0.3 prompt suiteはmean decode `33.612 tok/s`、V620 controlled v0.3 prompt suiteはmean decode `20.476 tok/s` だった。
- RDNA4のrow groupingはまだ局所最適を探索しておらず、またAQ4 matvec/add以外のvariantは初回HIPRTC compileがprefillへ混ざる可能性が残っていた。

### 今回の変更点

- RDNA4/gfx12のAQ4 matvec系 `ULLM_AQ4_ROWS_PER_BLOCK` を `4` から `16` に変更した。RDNA2など `gfx12` 以外は引き続き `1`。
- 探索結果はR9700 prompt16/gen128で、rows4 `34.637 tok/s`、rows8 `36.727 tok/s`、rows16 `36.948 tok/s`、rows32 `30.241 tok/s`。rows16を採用した。
- resident linear-attn load時にAQ4 `matvec`、`matvec_gate_beta`、`matvec_silu_mul` も一度だけprewarmするようにした。既存の `matvec_add` prewarmと合わせて、初回prefillにAQ4 variantのHIPRTC compileが混ざる問題を減らした。
- code commitは `b8182eb Use wider RDNA4 AQ4 row grouping`。
- commit後のR9700 prompt16/gen128では生成token列がrows4 ctrows版と一致し、verified `true` だった。decode `36.995 tok/s`、skip2 `36.981 tok/s`、last8 `36.111 tok/s`、prefill `28.454 tok/s`。rows4 ctrows版比でdecode `+2.358 tok/s`、約 `+6.81%`、prefill `+3.385 tok/s`、約 `+13.50%`。
- commit後のV620/RDNA2 prompt16/gen32でも生成token列がrows4 ctrows版と一致し、verified `true` だった。decode `21.281 tok/s`、skip2 `21.274 tok/s`、last8 `21.172 tok/s`、prefill `18.805 tok/s`。decodeは実質中立、prefillは `+2.162 tok/s`、約 `+12.99%`。
- R9700 controlled v0.3 prompt suiteではmean decode `35.872 tok/s`、min `32.151 tok/s`、max `36.785 tok/s`、mean prefill `33.279 tok/s`、verified all `true` を得た。output healthは `ok=5`、`warn=1`、`not_evaluated=1`。warnは `throughput_checklist_direct` の `low_unique_token_ratio`。
- V620/RDNA2 controlled v0.3 prompt suiteではmean decode `20.464 tok/s`、min `18.885 tok/s`、max `20.825 tok/s`、mean prefill `21.218 tok/s`、verified all `true` を得た。output healthはR9700と同じく `ok=5`、`warn=1`、`not_evaluated=1`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseでprompt token、生成token、stop reason、verified、output status、top logitsが一致し、最大logit差はprefill `8.11e-6`、decode末尾 `1.41e-4` だった。
- R9700 rows16の層別同期診断gen32では、linear-attention 24層の平均が `0.758ms/layer`、合計 `18.203ms`、self-attention 8層の平均が `0.734ms/layer`、合計 `5.870ms`、lm_head平均が `3.367ms` だった。rows4 ctrows診断のlinear合計 `19.453ms`、self合計 `6.228ms` からさらに短縮した。

### 次の行動

1. AQ4フォーマット自体の壁にはまだ当たっていない。RDNA4ではrows16 groupingだけでR9700 controlled suite平均が `35.9 tok/s` まで伸びた。
2. RDNA2/RDNA4プロトタイプとしては、R9700 `35.9 tok/s`、V620 `20.5 tok/s`、cross-device生成token一致まで確認できたため、速度と明白な品質崩壊チェックは十分な水準に到達したと考える。
3. 既存高速量子化engine級の `~79 tok/s` を目指すなら、次は単一AQ4 matvecのrows調整ではなく、linear-attn/MLP projection群の大きな融合が必要。rows32で悪化したため、これ以上rows_per_blockだけを増やす方向は一旦止める。

## 2026-07-06 RDNA4 AQ4 launch grid alignment

### 前回の要点

- RDNA4 rows16 grouping + AQ4 variant prewarm後のR9700 prompt16/gen128はdecode `36.995 tok/s`、V620 prompt16/gen32はdecode `21.281 tok/s` だった。
- R9700 controlled v0.3 prompt suiteはmean decode `35.872 tok/s`、mean prefill `33.279 tok/s`、verified all `true` だった。
- ただしkernel source側の `ULLM_AQ4_ROWS_PER_BLOCK=16` に対して、launch grid計算側の `Aq4MatvecLaunchConfig.rows_per_block` が `4` のまま残っており、RDNA4では必要以上の空blockをlaunchしていた。

### 今回の変更点

- RDNA4/gfx12のAQ4 matvec launch configを `rows_per_block=4` から `16` に揃えた。kernel sourceのcompile-time rows_per_blockとgrid計算を一致させた。
- code commitは `b27c888 Align AQ4 RDNA4 launch row grouping`。
- R9700 prompt16/gen128では生成token列が前段rows16版と一致し、verified `true` だった。decode `37.207 tok/s`、skip2 `37.194 tok/s`、last8 `36.294 tok/s`、prefill `28.719 tok/s` を得た。前段比でdecode `+0.213 tok/s`、約 `+0.58%`。
- V620/RDNA2 prompt16/gen32では生成token列が前段rows16版と一致し、verified `true` だった。decode `21.230 tok/s`、skip2 `21.223 tok/s`、last8 `21.154 tok/s`。RDNA2 pathはコード上変わっていないため、差は単発ノイズ幅と判断した。
- R9700 controlled v0.3 prompt suiteではmean decode `36.130 tok/s`、min `32.392 tok/s`、max `36.941 tok/s`、mean prefill `32.680 tok/s`、verified all `true` を得た。前段suite比でmean decode `+0.257 tok/s`、約 `+0.72%`。output healthは `ok=5`、`warn=1`、`not_evaluated=1` のまま。
- 新R9700 suiteと既存V620 rows16 prewarm suiteのcross-device guardは `atol=1e-3` でpass。7/7 caseでprompt token、生成token、stop reason、verified、output status、top logitsが一致し、最大logit差はprefill `8.11e-6`、decode末尾 `1.41e-4` だった。

### 次の行動

1. rows_per_blockまわりの明確な実装不一致は解消した。ここから先の単純なrow grouping調整は収穫逓減に入っている。
2. R9700 controlled suiteは平均 `36.1 tok/s` まで上がったが、既存高速量子化engine級の `~79 tok/s` との差はまだ大きい。
3. 次の改善候補はlinear-attn/MLP projection群の融合、またはその前段としてprojection単位のGPU event/同期診断を追加して、どのAQ4 matvec群をまとめるべきかを定量化すること。

## 2026-07-06 linear-attn component timing diagnostic

### 前回の要点

- RDNA4 rows16 launch grid alignment後のR9700 prompt16/gen128はdecode `37.207 tok/s`、controlled v0.3 prompt suiteはmean decode `36.130 tok/s` だった。
- rows_per_blockまわりの明確な不一致は解消済みで、次の候補はlinear-attn/MLP projection群の融合だった。
- ただし融合対象を決めるには、linear-attn層内でAQ4 projection、補助kernel、recurrent stepのどれが支配的かを分解する必要があった。

### 今回の変更点

- `ULLM_SYNC_LINEAR_ATTN_COMPONENTS_FOR_TIMING=1` 診断モードを追加し、linear-attn resident step内のcomponentごとに同期付き時間をJSONへ出せるようにした。
- 通常実行では追加同期を入れず、`decode.linear_attn_component_step_ms` はlinear-attn層だけcomponent object、self-attn層は `null` になる。
- R9700 prompt16/gen32の通常benchではdecode `38.068 tok/s`、skip2 `38.064 tok/s`、last8 `37.843 tok/s`、verified `true` を得た。診断コードは通常パスのtoken/sを落としていない。
- R9700 prompt16/gen32のcomponent同期診断ではdecode `30.259 tok/s`、skip2 `30.267 tok/s`、last8 `30.297 tok/s`、verified `true` を得た。これはcomponentごとの同期コスト込みなので実性能ではなく内訳診断として扱う。
- component同期診断の層別平均は、linear-attention 24層が `0.997ms/layer`、self-attention 8層が `0.715ms/layer`、layer stack平均が `29.690ms/token`、lm_head平均が `3.355ms/token` だった。
- linear-attn 1層平均 `0.996ms` の内訳は、AQ4 projection群が `0.778ms`、非AQ4補助処理が `0.218ms`。component平均は `mlp_gate_up_activation=0.262ms`、`mlp_down_residual=0.152ms`、`qkv_projection=0.117ms`、`gate_beta_projection=0.084ms`、`out_projection_residual=0.082ms`、`z_projection=0.080ms`、`recurrent=0.079ms`。

### 次の行動

1. ここでもAQ4フォーマット自体の壁にはまだ当たっていない。支配要因は、同じ正規化済み入力に対して複数のAQ4 projectionを順にlaunchしている構成だった。
2. 次に効果が出そうなのは、linear-attn入力側の `qkv`、`z`、`a/b gate-beta` をまとめる特殊AQ4 projection kernel。特に `a/b gate-beta` は出力が小さい割に `0.084ms/layer` で、固定費削減の余地がある。
3. MLP側は `gate/up` と `down` が最大だが、`down` はactivation依存で同一kernel化しにくい。先に入力側projection融合でlaunch数と入力read pathを減らし、token/sと生成token列一致を確認する。

### 棄却した探索

- gate-betaだけRDNA4 rows_per_blockを `16 -> 32` にする実験は悪化した。R9700 prompt16/gen32通常benchは `38.068 tok/s -> 36.507 tok/s`、component同期診断の `gate_beta_projection_ms` は `0.084ms -> 0.132ms` になった。出力token列は一致したが速度が悪化したため、runtime差分と実験artifactは残していない。
- `qkv`、`z`、`a/b gate-beta` を別HIP streamへ投げるparallel projection実験も悪化した。R9700 prompt16/gen32はenv off `38.049 tok/s`、env on `37.363 tok/s` で、生成token列は一致したが `-0.686 tok/s`。input RMSNorm後のhost同期がGPU queueの非同期性を壊すため、event待ちまたは単一融合kernelなしのstream分割は採用しない。

## 2026-07-06 AQ4 qkv/z pair matvec fusion

### 前回の要点

- linear-attn component timing診断では、linear-attn 1層平均 `0.996ms` のうちAQ4 projection群が `0.778ms` を占めていた。
- 単純なgate-beta rows32化と別HIP stream投入はどちらも悪化したため、host同期やlaunch数を増やす方向ではなく、同じ正規化済み入力を読むprojectionを単一kernelへまとめる方向を優先した。
- 最初の対象はlinear-attn入力側の `qkv` と `z`。`a/b gate-beta` はpost処理が違うため、まず素直な2出力matvec pairで効果と安全性を確認した。

### 今回の変更点

- runtime ABIに `ullm_runtime_aq4_matvec_pair_f32` を追加し、HIP pathでは1回のkernel launchで左行列と右行列のAQ4 matvecを同じ入力から計算できるようにした。CPU pathは既存host matvecを2回呼ぶ参照実装にしている。
- Rust wrapperに `aq4_matvec_pair_f32`、engine側に `PackageAq4ResidentMatvec::matvec_pair_with` を追加し、linear-attnの `qkv` と `z` projectionでデフォルト使用するようにした。`ULLM_DISABLE_AQ4_MATVEC_PAIR_QKV_Z=1` で無効化でき、`ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL=1` でfallback混入を検出できる。
- resident linear-attn load時に `qkv/z` pair kernelもprewarmするようにし、新規HIPRTC compileがprefill測定へ混ざる問題を減らした。
- CPU/HIP単体テストとして、左2行・右1行を同じ入力から同時に計算する最小ケースを追加した。
- R9700 prompt16/gen32のoff/on比較では、off `38.085 tok/s`、on `38.519 tok/s` で、pair有効化により `+0.434 tok/s`。生成token列は一致した。
- V620 prompt16/gen32のoff/on比較では、off `21.285 tok/s`、on `21.420 tok/s` で、pair有効化により `+0.135 tok/s`。生成token列は一致した。
- デフォルト有効後のR9700 prompt16/gen128ではverified `true`、decode `37.762 tok/s`、skip2 `37.748 tok/s`、last8 `36.941 tok/s`、prefill `29.014 tok/s` を得た。前段rows16 grid alignmentの単発 `37.207 tok/s` から `+0.554 tok/s`。
- デフォルト有効後のV620 prompt16/gen32ではverified `true`、decode `21.365 tok/s`、skip2 `21.355 tok/s`、last8 `21.272 tok/s`、prefill `18.963 tok/s` を得た。前段rows16 grid alignmentの単発 `21.230 tok/s` から `+0.134 tok/s`。
- R9700 controlled v0.3 prompt suiteではmean decode `37.191 tok/s`、min `33.232 tok/s`、max `38.176 tok/s`、mean prefill `33.150 tok/s`、verified all `true`。前段suite平均 `36.130 tok/s` から `+1.061 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `20.866 tok/s`、min `19.255 tok/s`、max `21.207 tok/s`、mean prefill `21.297 tok/s`、verified all `true`。前段suite平均 `20.464 tok/s` から `+0.403 tok/s`。
- suite output healthはR9700/V620とも `ok=5`、`warn=1`、`not_evaluated=1`。warnは `low_unique_token_ratio` で、速度変更による明白な品質崩壊は観測していない。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseでprompt token、生成token、stop reason、verified、output status、top logitsが一致し、最大logit差はprefill `2.86e-6`、decode末尾 `2.00e-5` だった。

### 次の行動

1. qkv/z pair fusionは小さいが一貫して効いた。AQ4フォーマット自体の壁ではなく、projectionごとのlaunch/read path固定費がまだ支配的だと考える。
2. R9700 controlled suite平均は `37.0 tok/s` まで上がったが、既存高速量子化engine級の `~79 tok/s` には届かない。単一pairだけでは不足しており、より大きいprojection束をまとめる必要がある。
3. 次は `qkv`、`z`、`a/b gate-beta` を同一kernel内で扱うか、MLPの `gate/up` 側を2出力以上でまとめる。ただし品質guardは維持し、token/sと出力healthを同時に見る。

## 2026-07-06 AQ4 qkv/z/gate-beta fused projection

### 前回の要点

- qkv/z pair fusion後のR9700 prompt16/gen128はdecode `37.762 tok/s`、controlled suite平均は `37.191 tok/s` だった。
- V620 controlled suite平均は `20.866 tok/s` で、RDNA2側はpair fusionが小幅改善からほぼ中立の範囲だった。
- まだ既存高速量子化engine級には届かないため、同じinput RMSNorm出力を読む `qkv`、`z`、`a/b gate-beta` をさらにまとめる余地があった。

### 今回の変更点

- runtime ABIに `ullm_runtime_aq4_matvec_qkv_z_gate_beta_f32` を追加し、HIP pathでは `qkv`、`z`、`a/b gate-beta` を1回のrow-reduction kernelで処理できるようにした。CPU pathは既存host matvecとgate-beta host関数を使う参照実装にした。
- Rust wrapperとengine側の `PackageAq4ResidentMatvec::matvec_qkv_z_gate_beta_with` を追加し、失敗時は既存のqkv/z pair + gate-betaへfallbackするようにした。`ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL=1` ではfallbackせず検出する。
- V620/RDNA2ではfusion onがoffよりわずかに遅かったため、正式デフォルトはHIP compute major `>= 12`、つまりRDNA4系だけfusion有効にした。RDNA2は既存のqkv/z pair + gate-betaを維持する。
- CPU単体テストとして、qkv 2行、z 1行、a/b 1 headを同じ入力から計算し、gate/beta変換まで確認する最小ケースを追加した。
- R9700 prompt16/gen32のoff/on比較では、off `38.581 tok/s`、on `39.051 tok/s` で、fusionにより `+0.470 tok/s`。生成token列は一致した。
- V620 prompt16/gen32のoff/on比較では、off `21.432 tok/s`、on `21.405 tok/s` で、fusionは `-0.027 tok/s`。生成token列は一致したが、正式デフォルトからは外した。
- RDNA4限定デフォルト後のR9700 prompt16/gen128ではverified `true`、fusion `true`、decode `38.242 tok/s`、skip2 `38.228 tok/s`、last8 `37.296 tok/s`、prefill `29.427 tok/s` を得た。qkv/z pair版の単発 `37.762 tok/s` から `+0.480 tok/s`。
- RDNA4限定デフォルト後のV620 prompt16/gen32ではverified `true`、fusion `false`、decode `21.367 tok/s`、skip2 `21.359 tok/s`、last8 `21.278 tok/s`、prefill `18.973 tok/s` を得た。qkv/z pair版と生成token列は一致し、速度も実質同等だった。
- R9700 controlled v0.3 prompt suiteではmean decode `37.705 tok/s`、min `33.620 tok/s`、max `38.575 tok/s`、mean prefill `34.149 tok/s`、verified all `true`。qkv/z pair suite平均 `37.191 tok/s` から `+0.515 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `20.859 tok/s`、min `19.243 tok/s`、max `21.242 tok/s`、mean prefill `21.182 tok/s`、verified all `true`。qkv/z pair suite平均 `20.866 tok/s` と実質同等。
- suite output healthはR9700/V620とも `ok=5`、`warn=1`、`not_evaluated=1`。warnは従来同様 `low_unique_token_ratio` で、速度変更による明白な品質崩壊は観測していない。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseでprompt token、生成token、stop reason、verified、output status、top logitsが一致し、最大logit差はprefill `1.91e-6`、decode末尾 `5.67e-5` だった。

### 次の行動

1. qkv/z/gate-beta融合も小さいがR9700で一貫して効いた。AQ4フォーマット自体の壁ではなく、linear-attn内のprojection launch/read path固定費がまだ詰められる。
2. ただしR9700 controlled suite平均は `37.7 tok/s` で、既存高速量子化engine級の `~79 tok/s` にはまだ遠い。ここから大きく伸ばすには、projection単位の小融合だけでなく、qkv_prepare/recurrent/post周辺まで含めた層内融合か、self-attn側q/k/vの同時projectionが必要だと考える。
3. 次の候補は、self-attn q/k/v projectionのpair/triple化、またはlinear-attnのqkv projectionとqkv_prepareの接続を再設計すること。V620はRDNA4向け融合を盲目的に有効化しない方針を維持する。

## 2026-07-06 self-attn AQ4 q/k pair projection

### 前回の要点

- qkv/z/gate-beta融合後のR9700 controlled suite平均は `37.705 tok/s`、V620 controlled suite平均は `20.859 tok/s` だった。
- linear-attn側の入力projection融合はR9700で一貫して効いたが、既存高速量子化engine級の `~79 tok/s` にはまだ届かなかった。
- 次の低リスク候補は、self-attn層で同じinput RMSNorm出力を読む `q_proj` と `k_proj` を既存AQ4 pair kernelへ載せることだった。

### 今回の変更点

- self-attn resident stepの `q_proj` と `k_proj` を、既存 `aq4_matvec_pair_f32` 経由の1 kernel launchへまとめた。`v_proj` は従来通り単独AQ4 matvec。
- `ULLM_DISABLE_AQ4_MATVEC_PAIR_SELF_ATTN_QK=1` で旧経路へ戻せるようにし、decode reportへ `use_aq4_matvec_pair_self_attn_qk` を出すようにした。
- self-attn layer load時にq/k pair projectionをprewarmし、初回HIPRTC compileがprefillやdecode測定に混ざりにくくした。
- R9700 prompt16/gen32 off/on比較では、off `39.033 tok/s`、on `39.232 tok/s`。生成token列は一致した。
- V620 prompt16/gen32 off/on比較では、off `21.357 tok/s`、on `21.376 tok/s`。生成token列は一致した。
- R9700 prompt16/gen128 defaultではverified `true`、decode `38.445 tok/s`、skip2 `38.428 tok/s`、last8 `37.433 tok/s`、prefill `29.563 tok/s` を得た。前段default `38.242 tok/s` から `+0.203 tok/s`。
- V620 prompt16/gen32 defaultではverified `true`、decode `21.415 tok/s`、skip2 `21.409 tok/s`、last8 `21.331 tok/s`、prefill `18.095 tok/s` を得た。前段default `21.367 tok/s` から `+0.048 tok/s`。
- R9700 controlled v0.3 prompt suiteではmean decode `37.856 tok/s`、min `33.867 tok/s`、max `38.798 tok/s`、mean prefill `34.097 tok/s`、verified all `true`。前段suite平均 `37.705 tok/s` から `+0.150 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `20.889 tok/s`、min `19.269 tok/s`、max `21.248 tok/s`、mean prefill `21.251 tok/s`、verified all `true`。前段suite平均 `20.859 tok/s` から `+0.029 tok/s`。
- suite output healthはR9700/V620とも `ok=5`、`warn=1`、`not_evaluated=1`。warnは従来同様 `low_unique_token_ratio`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `1.91e-6`、decode末尾 `5.67e-5` だった。

### 次の行動

1. self-attn q/k pairは安全で正方向だが、suite平均ではR9700 `+0.4%` 程度に留まる。AQ4フォーマット自体の壁ではなく、launch/read path固定費の小削減に近い。
2. 既存高速量子化engine級との差をさらに詰めるなら、self-attn `q/k/v` の真の3出力kernel、linear-attnのprojection+prepare接続、またはrecurrent/postを含む層内大融合が必要。
3. 今回の改善でR9700 suiteは `37.9 tok/s`、V620 suiteは `20.9 tok/s`。RDNA4/RDNA2で動くプロトタイプ水準は維持しつつ、次は小さなpair化より大きい融合単位を優先する。

## 2026-07-06 self-attn AQ4 q/k/v triple projection

### 前回の要点

- self-attn q/k pair後のR9700 controlled suite平均は `37.856 tok/s`、V620 controlled suite平均は `20.889 tok/s` だった。
- q/k pairは安全で正方向だったが、self-attn層内ではまだ `v_proj` が単独launchとして残っていた。
- 次の候補はruntime ABIとして3出力AQ4 matvecを追加し、self-attn `q/k/v` を1 kernelへまとめることだった。

### 今回の変更点

- runtime ABIに `ullm_runtime_aq4_matvec_triple_f32`、Rust wrapperに `aq4_matvec_triple_f32` を追加した。
- HIPRTC kernel `ullm_aq4_matvec_triple_f32_kernel` を追加し、3つのAQ4 matrixを同じinputから1回のrow-reduction launchで計算するようにした。CPU pathはhost matvecを3回呼ぶ参照実装。
- `PackageAq4ResidentMatvec::matvec_triple_with` とprewarmを追加し、self-attn resident stepの `q_proj/k_proj/v_proj` をデフォルトでtriple化した。
- `ULLM_DISABLE_AQ4_MATVEC_TRIPLE_SELF_ATTN_QKV=1` でtripleを無効化でき、無効時は既存q/k pair + v単独へfallbackする。
- prompt benchのrequired HIP kernel listへ `ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL=1` を追加し、suite測定でtriple fallback混入を検出できるようにした。
- runtime-sysのCPU/HIP単体テストを追加し、`cargo test -p ullm-runtime-sys -- --test-threads=1` は78 tests pass。
- R9700 prompt16/gen32 off/on比較では、triple off `39.257 tok/s`、on `39.760 tok/s`。生成token列は一致した。
- V620 prompt16/gen32 off/on比較では、triple off `21.408 tok/s`、on `21.411 tok/s`。生成token列は一致し、速度は中立だった。
- R9700 prompt16/gen128 defaultではverified `true`、decode `38.836 tok/s`、skip2 `38.817 tok/s`、last8 `37.756 tok/s`、prefill `29.655 tok/s` を得た。q/k pair default比で `+0.390 tok/s`。
- V620 prompt16/gen32 defaultではverified `true`、decode `21.414 tok/s`、skip2 `21.407 tok/s`、last8 `21.335 tok/s`、prefill `18.962 tok/s` を得た。q/k pair default比では実質中立。
- R9700 controlled v0.3 prompt suiteではmean decode `38.257 tok/s`、min `34.096 tok/s`、max `39.295 tok/s`、mean prefill `34.489 tok/s`、verified all `true`。q/k pair suite平均 `37.856 tok/s` から `+0.402 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `20.930 tok/s`、min `19.290 tok/s`、max `21.305 tok/s`、mean prefill `21.110 tok/s`、verified all `true`。q/k pair suite平均 `20.889 tok/s` から `+0.041 tok/s`。
- suite output healthはR9700/V620とも `ok=5`、`warn=1`、`not_evaluated=1`。warnは従来同様 `low_unique_token_ratio`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `1.91e-6`、decode末尾 `5.67e-5` だった。

### 次の行動

1. q/k/v tripleはR9700で明確に効き、V620でも中立なので正式採用できる。ここでもAQ4フォーマット自体の壁ではなく、projection launch固定費を削れている。
2. R9700 suite平均は `38.3 tok/s` まで上がったが、既存高速量子化engine級の `~79 tok/s` にはまだ遠い。
3. 次の改善候補は、linear-attn側のMLP gate/up fused matvecのRDNA4向け行grouping再探索、またはlinear-attn projection+prepare/recurrent/postをまたぐ大きい融合。self-attn側の単純projection融合は収穫逓減に近づいている。

### 棄却した探索

- RDNA4/gfx12の `aq4_matvec_silu_mul` だけ `rows_per_block=32` にする実験は悪化した。R9700 prompt16/gen32はq/k/v triple基準 `39.760 tok/s` から `38.306 tok/s` へ低下した。生成はverified trueだが速度面で不採用。
- 同kernelだけ `rows_per_block=8` にする実験は速度だけなら大きく改善した。R9700 prompt16/gen32は `46.151 tok/s`、prefill `33.618 tok/s` まで上がった。しかしR9700 prompt16/gen128で生成token列が9 token目から分岐し、`24` の反復に落ちた。q/k/v triple基準の9 token目は `30226` で、rows8版は `24`。token/sと引き換えに出力品質が壊れるため不採用。
- rows8/rows32実験後、runtime sourceは既定のRDNA4 `rows_per_block=16` へ戻した。復帰確認のR9700 prompt16/gen32は `39.617 tok/s`、生成token列はq/k/v triple基準と一致した。

## 2026-07-06 AQ4 SiLU-mul one-pass input reuse

### 前回の要点

- self-attn q/k/v triple後のR9700 controlled suite平均は `38.257 tok/s`、V620 controlled suite平均は `20.930 tok/s` だった。
- sync each layer計測では、linear-attn層合計がまだ支配的だった。過去のcomponent timingでは `MLP gate/up activation` がlinear-attn 1層あたり約 `0.262ms` と大きい。
- 直前の `aq4_matvec_silu_mul rows8` 実験は高速に見えたが、source側の `rows_per_block=8` とhost側launch gridの `rows_per_block=16` が一致していなかった可能性が高く、無効な計測として扱う必要があった。

### 今回の変更点

- `aq4_matvec_silu_mul` のrows8を、sourceとhost launch configを一致させた有効な形で再検証した。R9700 prompt16/gen128では生成token列はq/k/v triple基準と一致したが、decodeは `38.836 -> 37.973 tok/s` に低下したため不採用。
- 以前の無効rows8で観測した `44.936 tok/s` とtoken崩壊は、行欠けによる不正な高速化だったと判断した。
- 正式実装として、HIPRTCの `ullm_aq4_matvec_silu_mul_f32_kernel` に gate/up のone-pass helperを追加した。
- gate/upの `group_size` が同じで `cols % group_size == 0` の通常ケースでは、同じinputベクトルを1回だけ走査し、gate raw sumとup raw sumを同時に計算する。group sizeが異なる場合や割り切れない場合は従来の2回走査経路へfallbackする。
- 数値順序は各matrixについて従来と同じgroup順を保つため、単体・prompt guardではtoken/logit一致を維持した。
- R9700 prompt16/gen32では、q/k/v triple基準 `39.760 tok/s` から `42.423 tok/s` へ改善。生成token列とlast top logitは一致。
- V620 prompt16/gen32では、q/k/v triple基準 `21.411 tok/s` から `24.589 tok/s` へ改善。生成token列とlast top logitは一致。
- R9700 prompt16/gen128では、q/k/v triple基準 `38.836 tok/s` から `41.407 tok/s` へ改善。生成token列とlast top logitは一致。
- R9700 controlled v0.3 prompt suiteではmean decode `40.810 tok/s`、min `36.158 tok/s`、max `41.910 tok/s`、mean prefill `36.555 tok/s`、verified all `true`。q/k/v triple suite平均 `38.257 tok/s` から `+2.553 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `23.890 tok/s`、min `21.788 tok/s`、max `24.372 tok/s`、mean prefill `24.367 tok/s`、verified all `true`。q/k/v triple suite平均 `20.930 tok/s` から `+2.960 tok/s`。
- suite output healthはR9700/V620とも `ok=5`、`warn=1`、`not_evaluated=1`。warnは従来同様 `low_unique_token_ratio`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `1.91e-6`、decode末尾 `5.67e-5` だった。

### 次の行動

1. one-pass化はAQ4フォーマットの方針に沿っており、FP32展開ではなくraw-value/codebook/scaleをkernel内で同時処理する改善として正式採用できる。
2. R9700 suite平均は `40.8 tok/s` まで上がった。まだ既存高速量子化engine級の `~79 tok/s` には遠いが、今回の改善はAQ4固有dequant経路の重複メモリアクセス削減として有効。
3. 次は `aq4_matvec_qkv_z_gate_beta` や `aq4_matvec_pair/triple` 側でも同様にinput再利用・group loop共有が可能かを見る。ただし複数matrixのgroup size差やrow layout差があるため、guardを維持して段階的に進める。

## 2026-07-06 AQ4 gate-beta one-pass input reuse

### 前回の要点

- AQ4 SiLU-mul one-pass化後のR9700 controlled suite平均は `40.810 tok/s`、V620 controlled suite平均は `23.890 tok/s` だった。
- 改善の本質は、AQ4をFP32に展開せず、raw index/codebook/scaleをkernel内で扱いながら同じinputベクトルの読みを共有することだった。
- 同じ構造はlinear-attnの `a/b gate-beta` にもあり、R9700では `qkv/z/gate-beta` fused kernel内のhead部分、V620では単独 `aq4_matvec_gate_beta` kernelで使われていた。

### 今回の変更点

- HIPRTCの `ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel` に、a/b head行をone-passで計算するhelperを追加した。
- HIPRTCの `ullm_aq4_matvec_gate_beta_f32_kernel` にも同等のone-pass helperを追加した。
- `a_group_size == b_group_size` かつ `cols % group_size == 0` の通常ケースでは、同じinput走査でa/b raw sumを同時に計算する。条件を満たさない場合は従来の2回走査経路へfallbackする。
- R9700 prompt16/gen32では、SiLU-mul one-pass基準 `42.423 tok/s` から `43.176 tok/s` へ改善。生成token列とlast top logitは一致。
- V620 prompt16/gen32では、SiLU-mul one-pass基準 `24.589 tok/s` から `24.549 tok/s` で誤差程度。生成token列とlast top logitは一致。
- R9700 prompt16/gen128では、SiLU-mul one-pass基準 `41.407 tok/s` から `42.090 tok/s` へ改善。生成token列とlast top logitは一致。
- R9700 controlled v0.3 prompt suiteではmean decode `41.605 tok/s`、min `36.693 tok/s`、max `42.730 tok/s`、mean prefill `37.461 tok/s`、verified all `true`。SiLU-mul one-pass suite平均 `40.810 tok/s` から `+0.794 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `23.896 tok/s`、min `21.823 tok/s`、max `24.369 tok/s`、mean prefill `24.367 tok/s`、verified all `true`。SiLU-mul one-pass suite平均 `23.890 tok/s` と実質同等。
- suite output healthはR9700/V620とも `ok=5`、`warn=1`、`not_evaluated=1`。warnは従来同様 `low_unique_token_ratio`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `1.91e-6`、decode末尾 `5.67e-5` だった。

### 次の行動

1. gate-beta one-pass化はR9700で明確に効き、V620で中立なので正式採用できる。
2. R9700 suite平均は `41.6 tok/s`、V620 suite平均は `23.9 tok/s`。AQ4フォーマット自体の壁にはまだ達しておらず、kernel内のinput再利用で引き続き改善できている。
3. `pair/triple` の単純横展開は、現在のrow-concat layoutだと同じthread groupが複数matrixの同じ行を同時計算していないため、そのままone-pass化できない。次に進めるなら、row-paired layoutの新kernelか、linear-attnのprepare/recurrent/post周辺の層内融合を検討する。

## 2026-07-06 AQ4 pair/triple row-paired layout

### 前回の要点

- gate-beta one-pass化後のR9700 controlled suite平均は `41.605 tok/s`、V620 controlled suite平均は `23.896 tok/s` だった。
- `aq4_matvec_pair` / `aq4_matvec_triple` は、複数matrixを1 launchへまとめていたが、内部では出力行を単純連結していた。そのため同じ行番号の複数matrixが同じinputを別々に走査していた。
- pair/tripleをさらに速くするには、row-concatではなくrow-paired layoutにして、同じ行番号を同じthread groupで計算する必要があった。

### 今回の変更点

- HIPRTCの `ullm_aq4_matvec_pair_f32_kernel` をrow-paired layoutへ変更した。作業行数は `left_rows + right_rows` ではなく `max(left_rows, right_rows)` とし、同じ行番号が両matrixに存在する場合は1回のinput走査でleft/rightを同時計算する。
- HIPRTCの `ullm_aq4_matvec_triple_f32_kernel` も同様に、作業行数を3 matrixの `max(rows)` に変更した。3 matrixすべてに同じ行番号が存在し、group sizeが一致する通常ケースでは1回のinput走査でfirst/second/thirdを同時計算する。
- group sizeが一致しない場合や、あるmatrixに行が存在しない場合は従来の単独thread sum経路へfallbackする。
- host側grid計算も `sum(rows)` から `max(rows)` に変更し、sourceとlaunchの作業行数を一致させた。
- R9700 prompt16/gen32では、gate-beta one-pass基準 `43.176 tok/s` から `43.301 tok/s` へ小幅改善。生成token列とlast top logitは一致。
- V620 prompt16/gen32では、gate-beta one-pass基準 `24.549 tok/s` から `25.487 tok/s` へ改善。生成token列とlast top logitは一致。
- R9700 prompt16/gen128では、gate-beta one-pass基準 `42.090 tok/s` から `42.125 tok/s` へほぼ中立。生成token列とlast top logitは一致。
- R9700 controlled v0.3 prompt suiteではmean decode `41.669 tok/s`、min `36.777 tok/s`、max `42.690 tok/s`、mean prefill `37.537 tok/s`、verified all `true`。gate-beta one-pass suite平均 `41.605 tok/s` から `+0.064 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `24.780 tok/s`、min `22.533 tok/s`、max `25.289 tok/s`、mean prefill `25.434 tok/s`、verified all `true`。gate-beta one-pass suite平均 `23.896 tok/s` から `+0.884 tok/s`。
- q/k/v triple基準から見ると、R9700 suite平均は `38.257 -> 41.669 tok/s`、V620 suite平均は `20.930 -> 24.780 tok/s` まで上がった。
- suite output healthはR9700/V620とも `ok=5`、`warn=1`、`not_evaluated=1`。warnは従来同様 `low_unique_token_ratio`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `1.91e-6`、decode末尾 `5.67e-5` だった。

### 次の行動

1. row-paired layoutはV620/RDNA2で特に効き、R9700/RDNA4でも中立から小幅改善なので正式採用できる。
2. R9700 controlled suite平均は `41.7 tok/s`、V620 controlled suite平均は `24.8 tok/s`。まだ既存高速量子化engine級の `~79 tok/s` には遠い。
3. 単純なAQ4 matvec内input再利用はかなり拾った。次に大きく伸ばすには、linear-attnの `qkv_prepare + recurrent`、attention post、またはMLP downを含む層内融合が必要になる可能性が高い。

## 2026-07-06 AQ4 qkv/z row-paired layout

### 前回の要点

- pair/triple row-paired化後のR9700 controlled suite平均は `41.669 tok/s`、V620 controlled suite平均は `24.780 tok/s` だった。
- `aq4_matvec_qkv_z_gate_beta` fused kernelでは、R9700/RDNA4向けにqkv/z/gate-betaを1 launchへまとめていたが、qkv行とz行は内部で連結配置され、同じinputを別々に走査していた。
- gate-beta head側は既にone-pass化済みだったため、残る局所改善候補はqkv/z projection側のrow-paired化だった。

### 今回の変更点

- HIPRTCの `ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel` で、projection作業行数を `qkv_rows + z_rows` から `max(qkv_rows, z_rows)` へ変更した。
- 同じ行番号がqkv/z双方に存在し、group sizeが一致して `cols` を割り切れる通常ケースでは、1回のinput走査でqkv raw sumとz raw sumを同時計算する。
- group sizeが一致しない場合や片側に行が存在しない場合は、従来と同じ単独thread sumへfallbackする。
- host側grid計算も `max(qkv_rows, z_rows) + heads` に変更し、kernel source側の作業行数と一致させた。
- R9700 prompt16/gen32では、pair/triple row-paired基準 `43.301 tok/s` から `43.815 tok/s` へ改善。生成token列とlast top logitは一致。
- V620 prompt16/gen32では、pair/triple row-paired基準 `25.487 tok/s` から `25.471 tok/s` で中立。生成token列とlast top logitは一致。
- R9700 prompt16/gen128では、pair/triple row-paired基準 `42.125 tok/s` から `42.952 tok/s` へ改善。生成token列とlast top logitは一致。
- R9700 controlled v0.3 prompt suiteではmean decode `42.306 tok/s`、min `37.282 tok/s`、max `43.438 tok/s`、mean prefill `37.797 tok/s`、verified all `true`。pair/triple row-paired suite平均 `41.669 tok/s` から `+0.637 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `24.776 tok/s`、min `22.591 tok/s`、max `25.283 tok/s`、mean prefill `25.387 tok/s`、verified all `true`。pair/triple row-paired suite平均 `24.780 tok/s` と実質同等。
- q/k/v triple基準から見ると、R9700 suite平均は `38.257 -> 42.306 tok/s`、V620 suite平均は `20.930 -> 24.776 tok/s` まで上がった。
- suite output healthはR9700/V620とも `ok=5`、`warn=1`、`not_evaluated=1`。warnは従来同様 `low_unique_token_ratio`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `1.91e-6`、decode末尾 `5.67e-5` だった。
- `cargo test -p ullm-runtime-sys -- --test-threads=1` は78 tests pass、`cargo test -p ullm-engine -- --test-threads=1` は97+14 tests pass、`cargo fmt --all --check` もpass。

### 次の行動

1. qkv/z row-paired化はR9700/RDNA4で追加の改善、V620/RDNA2で中立なので正式採用できる。
2. ここまでの改善はAQ4フォーマット自体の壁ではなく、同じinputロードとdequant処理の重複をkernel内で削る余地がまだあったことを示している。
3. R9700 controlled suite平均は `42.3 tok/s` で、既存高速量子化engine級の `~79 tok/s` にはまだ遠い。次は最新状態でcomponent timingを取り直し、残る律速がAQ4 matvec単体なのか、linear-attn recurrent/postやMLP down周辺なのかを切り分ける。

## 2026-07-06 AQ4 matvec-add paired idx4 load

### 前回の要点

- qkv/z row-paired化後のR9700 controlled suite平均は `42.306 tok/s`、V620 controlled suite平均は `24.776 tok/s` だった。
- 最新状態のR9700 sync-each-layer計測では、prompt16/gen32が `42.226 tok/s`、skip2が `42.208 tok/s` で、ウォームアップによる見かけの低下ではなかった。
- component timingは融合を外す診断モードだが、linear-attn層では `MLP gate/up + down` がcomponent時間の約 `39.2%`、qkv/z/gate-beta projectionが約 `28.1%` だった。
- 層別同期計測ではlinear-attn層とself-attn層がどちらも約 `0.63ms/layer` で、linear-attn固有処理だけでなく全層共通のMLP/AQ4 projectionが支配的になっていた。

### 今回の変更点

- HIPRTCの `ullm_aq4_matvec_add_thread_sum` で、`cols % group_size == 0` かつ偶数group sizeの通常ケースにpaired idx4 loadを追加した。
- 既存実装は隣接する偶数/奇数nibbleで同じpacked byteを2回読み得る形だった。新経路では1回のpacked byte loadでlow/high nibbleを順に取り出し、`input[col]` と `input[col + 1]` に掛ける。
- 加算順はlow nibble、high nibbleの順を保ち、奇数group sizeでは従来経路へfallbackする。
- 対象は `aq4_matvec_add` なので、linear-attn out projection residual、self-attn o projection residual、全層のMLP down residualに効く。
- R9700 prompt16/gen32では、qkv/z row-paired基準 `43.815 tok/s` から `47.109 tok/s` へ改善。生成token列は一致。
- V620 prompt16/gen32では、qkv/z row-paired基準 `25.471 tok/s` から `28.282 tok/s` へ改善。生成token列は一致。
- R9700 prompt16/gen128では、qkv/z row-paired基準 `42.952 tok/s` から `45.779 tok/s` へ改善。skip2は `45.754 tok/s` で、ウォームアップ除外後も改善が残った。
- R9700 controlled v0.3 prompt suiteではmean decode `45.338 tok/s`、min `39.285 tok/s`、max `46.718 tok/s`、mean prefill `40.037 tok/s`、verified all `true`。qkv/z row-paired suite平均 `42.306 tok/s` から `+3.032 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `27.524 tok/s`、min `24.788 tok/s`、max `28.142 tok/s`、mean prefill `28.450 tok/s`、verified all `true`。qkv/z row-paired suite平均 `24.776 tok/s` から `+2.749 tok/s`。
- suite output healthはR9700/V620とも `ok=5`、`warn=1`、`not_evaluated=1`。warnは従来同様 `low_unique_token_ratio`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `1.91e-6`、decode末尾 `5.67e-5` だった。
- `cargo test -p ullm-runtime-sys -- --test-threads=1` は78 tests pass、`cargo test -p ullm-engine -- --test-threads=1` は97+14 tests pass、`cargo fmt --all --check` もpass。

### 次の行動

1. paired idx4 loadはR9700/RDNA4とV620/RDNA2の両方で大きく効いたため正式採用できる。
2. これはAQ4フォーマット自体の壁ではなく、packed indexの読み方にまだ明確な低遅延化余地があったことを示している。
3. 次は同じpaired idx4 loadを `aq4_matvec_silu_mul`、`aq4_matvec_pair/triple`、`aq4_matvec_qkv_z_gate_beta` のone-pass helperにも広げられるかを見る。今回と同じく、token/sとtop logits guardを同時に見る必要がある。

## 2026-07-06 AQ4 projection paired idx4 load

### 前回の要点

- `aq4_matvec_add` のpaired idx4 load後、R9700 controlled suite平均は `45.338 tok/s`、V620 controlled suite平均は `27.524 tok/s` だった。
- 改善の本質は、AQ4のidx4 packed byteを偶数/奇数nibbleごとに読み直さず、1回のloadからlow/highを順に処理することだった。
- 同じ読み方はMLP gate/upの `silu_mul` と、self-attn/linear-attn projectionのpair/triple/qkv-z one-pass helperにも適用できる見込みだった。

### 今回の変更点

- `aq4_matvec_silu_mul` のgate/up one-pass helperにpaired idx4 loadを追加した。
- `aq4_matvec_pair`、`aq4_matvec_triple` のone-pass helperにpaired idx4 loadを追加した。
- `aq4_matvec_qkv_z_gate_beta` のqkv/z・a/b one-pass helperにpaired idx4 loadを追加した。
- qkv/z fused kernelではqkv側の余り行が単独sumに落ちるため、`ullm_aq4_qkv_z_gate_beta_thread_sum` にもpaired idx4 loadを追加した。
- low/high nibbleの処理順は維持し、奇数group sizeでは従来経路へfallbackする。
- R9700 prompt16/gen32では、matvec-add paired基準 `47.109 tok/s` から、silu-mul pairedで `49.229 tok/s`、projection pairedで `49.887 tok/s` へ改善。生成token列は一致。
- V620 prompt16/gen32では、matvec-add paired基準 `28.282 tok/s` から、silu-mul pairedで `30.882 tok/s`、projection pairedで `32.357 tok/s` へ改善。生成token列は一致。
- R9700 prompt16/gen128では、matvec-add paired基準 `45.779 tok/s` から projection pairedで `48.642 tok/s` へ改善。skip2は `48.619 tok/s` で、ウォームアップ除外後も改善が残った。
- R9700 controlled v0.3 prompt suiteではmean decode `48.177 tok/s`、min `41.740 tok/s`、max `49.686 tok/s`、mean prefill `42.124 tok/s`、verified all `true`。matvec-add paired suite平均 `45.338 tok/s` から `+2.839 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `31.374 tok/s`、min `27.845 tok/s`、max `32.197 tok/s`、mean prefill `33.172 tok/s`、verified all `true`。matvec-add paired suite平均 `27.524 tok/s` から `+3.850 tok/s`。
- qkv/z row-paired基準から見ると、R9700 suite平均は `42.306 -> 48.177 tok/s`、V620 suite平均は `24.776 -> 31.374 tok/s` まで上がった。
- suite output healthはR9700/V620とも `ok=5`、`warn=1`、`not_evaluated=1`。warnは従来同様 `low_unique_token_ratio`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `1.91e-6`、decode末尾 `5.67e-5` だった。
- `cargo test -p ullm-runtime-sys -- --test-threads=1` は78 tests pass、`cargo test -p ullm-engine -- --test-threads=1` は97+14 tests pass、`cargo fmt --all --check` もpass。

### 次の行動

1. paired idx4 loadはAQ4 dequant pathの基本方針として正式採用できる。
2. R9700はprompt16/gen32でほぼ `50 tok/s`、controlled suite平均で `48.2 tok/s` まで来たが、既存高速量子化engine級の `~79 tok/s` にはまだ届かない。
3. 次は最新状態でcomponent/layer timingを取り直し、残る律速がAQ4 matvec内の未対応single pathなのか、lm-head/top1やkernel launch数なのか、あるいはAQ4フォーマットのランダムcodebook/scale参照そのものなのかを切り分ける。

## 2026-07-06 AQ4 gate-beta paired idx4 load

### 前回の要点

- projection paired idx4 load後のR9700 controlled suite平均は `48.177 tok/s`、V620 controlled suite平均は `31.374 tok/s` だった。
- 最新sync-each-layer計測では、R9700 prompt16/gen32が `47.660 tok/s`、skip2が `47.653 tok/s`。層処理合計は `17.605ms/step` で、qkv/z row-paired時の `20.310ms/step` から下がっていた。
- component timingではMLP gate/up + downは下がった一方、単独 `aq4_matvec_gate_beta` はpaired idx4 load未対応だった。

### 今回の変更点

- 単独 `aq4_matvec_gate_beta` kernelのsingle sum/helperにpaired idx4 loadを追加した。
- R9700 prompt16/gen32では、projection paired基準 `49.887 tok/s` から `50.307 tok/s` へ改善。生成token列は一致。
- V620 prompt16/gen32では、projection paired基準 `32.357 tok/s` から `32.912 tok/s` へ改善。生成token列は一致。
- R9700 prompt16/gen128では、projection paired基準 `48.642 tok/s` から `49.011 tok/s` へ改善。skip2は `48.986 tok/s`。
- R9700 controlled v0.3 prompt suiteではmean decode `48.540 tok/s`、min `42.020 tok/s`、max `50.042 tok/s`、mean prefill `42.946 tok/s`、verified all `true`。projection paired suite平均 `48.177 tok/s` から `+0.363 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `31.956 tok/s`、min `28.305 tok/s`、max `32.831 tok/s`、mean prefill `33.820 tok/s`、verified all `true`。projection paired suite平均 `31.374 tok/s` から `+0.582 tok/s`。
- qkv/z row-paired基準から見ると、R9700 suite平均は `42.306 -> 48.540 tok/s`、V620 suite平均は `24.776 -> 31.956 tok/s` まで上がった。
- suite output healthはR9700/V620とも `ok=5`、`warn=1`、`not_evaluated=1`。warnは従来同様 `low_unique_token_ratio`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `1.91e-6`、decode末尾 `5.67e-5` だった。
- `cargo test -p ullm-runtime-sys -- --test-threads=1` は78 tests pass、`cargo test -p ullm-engine -- --test-threads=1` は97+14 tests pass、`cargo fmt --all --check` もpass。

### 次の行動

1. 単独gate-beta paired idx4 loadは小幅だがR9700/V620両方で正方向なので正式採用できる。
2. R9700の短いpromptでは `50 tok/s` 前後に到達したが、controlled suite平均は `48.5 tok/s` で、まだ十分なtoken/sとは言い切れない。
3. 次は未対応の汎用 `aq4_matvec` single pathと、層外のlm-head/top1・kernel launch固定費を切り分ける。

## 2026-07-06 AQ4 group-size 8/16 unrolled idx4 paths

### 前回の要点

- gate-beta paired idx4 load後のR9700 controlled suite平均は `48.540 tok/s`、V620 controlled suite平均は `31.956 tok/s` だった。
- sync-each-layerではR9700 prompt16/gen32が `47.660 tok/s`、層処理合計 `17.611ms/step`、lm-head/top1 `3.367ms/step` だった。
- rpb8は `71.356 tok/s` と高速に見えたが、1 token目から生成が崩壊したため不採用。rpb32は生成を保ったが `32.438 tok/s` まで低下したため不採用。
- BF16 lm-head matvec block size 128はlm-head時間が `3.367ms -> 4.389ms` に悪化。block size 32は微改善程度で、AQ4本体の改善に比べると優先度が低い。

### 今回の変更点

- AQ4各matvecのgroup loopに、実パッケージで支配的な `group_size == 16` と `group_size == 8` の固定長unroll経路を追加した。
- 対象は `aq4_matvec_add`、`aq4_matvec_pair`、`aq4_matvec_triple`、`aq4_matvec_qkv_z_gate_beta`、`aq4_matvec_silu_mul`、`aq4_matvec_gate_beta` の主要single/helper path。
- 既存のpaired idx4 loadを維持したまま、group size 16/8では固定長loopにしてcompilerが展開できる形にした。その他の偶数group sizeと奇数group sizeは従来fallbackを維持。
- R9700 prompt16/gen32では、gate-beta paired基準 `50.307 tok/s` から `55.792 tok/s` へ改善。生成token列は一致。
- V620 prompt16/gen32では、gate-beta paired基準 `32.912 tok/s` から `35.314 tok/s` へ改善。生成token列は一致。
- R9700 prompt16/gen128では、gate-beta paired基準 `49.011 tok/s` から `53.870 tok/s` へ改善。skip2は `53.833 tok/s`。
- R9700 controlled v0.3 prompt suiteではmean decode `53.385 tok/s`、min `45.115 tok/s`、max `55.382 tok/s`、mean prefill `45.989 tok/s`、verified all `true`。gate-beta paired suite平均 `48.540 tok/s` から `+4.844 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `34.141 tok/s`、min `30.031 tok/s`、max `35.087 tok/s`、mean prefill `36.729 tok/s`、verified all `true`。gate-beta paired suite平均 `31.956 tok/s` から `+2.185 tok/s`。
- qkv/z row-paired基準から見ると、R9700 suite平均は `42.306 -> 53.385 tok/s`、V620 suite平均は `24.776 -> 34.141 tok/s` まで上がった。
- suite output healthはR9700/V620とも `ok=5`、`warn=1`、`not_evaluated=1`。warnは従来同様 `low_unique_token_ratio`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `1.91e-6`、decode末尾 `5.67e-5` だった。
- `cargo test -p ullm-runtime-sys -- --test-threads=1` は78 tests pass、`cargo test -p ullm-engine -- --test-threads=1` は97+14 tests pass、`cargo fmt --all --check` もpass。

### 次の行動

1. group size 8/16の固定長unrollは大きく効き、AQ4 dequant pathの正式方針にできる。
2. R9700の通常promptでは55 tok/s台、controlled suite平均では53.4 tok/sまで来た。まだ既存高速量子化engine級の `~79 tok/s` には届かないが、AQ4フォーマット自体の壁にはまだ達していない。
3. 次はsync timingを取り直し、層内AQ4がどこまで残ったか、lm-head/top1の3ms台が相対的にどれだけ支配的になったかを再評価する。

## 2026-07-06 AQ4 matvec-add RDNA4 row group tuning

### 前回の要点

- group size 8/16固定長unroll後のR9700 controlled suite平均は `53.385 tok/s`、V620 controlled suite平均は `34.141 tok/s` だった。
- R9700 sync-each-layer計測では、prompt16/gen32が `52.914 tok/s`、層処理合計が `15.523ms/step`、lm-head/top1が `3.372ms/step`、step全体が `18.899ms/step` だった。
- component timing診断では合計 `0.814ms/token` のうち、MLP gate/up + downが `0.273ms/token`、qkv/z/gate-beta projectionが `0.244ms/token` だった。
- global `rows_per_block=8` は `71.356 tok/s` と高速に見えたが、1 token目から生成が崩壊したため不採用。global `rows_per_block=32` は生成を保ったが `32.438 tok/s` まで低下したため不採用。
- BF16 lm-head matvec block size 128はlm-head時間を `3.367ms -> 4.389ms` に悪化させた。block size 32は微改善程度で、AQ4本体の改善に比べると優先度が低い。

### 今回の変更点

- globalなAQ4 row group設定はRDNA4 `16`、RDNA2 `1` のまま維持した。
- RDNA4/gfx12の `aq4_matvec_add` だけ `rows_per_block=8` を使うようにし、HIPRTC source側の `ULLM_AQ4_ROWS_PER_BLOCK` とhost側grid計算を一致させた。
- 対象を `aq4_matvec_add` に限定したため、MLP down residual、self-attn o projection residual、linear-attn output residualには効くが、qkv/zやsilu-mulのrow groupingは変更しない。
- R9700 prompt16/gen32では、unroll基準 `55.792 tok/s` から `56.950 tok/s` へ改善。skip2は `56.913 tok/s`、last8は `56.517 tok/s` で、ウォームアップ除外後も改善が残った。
- V620 prompt16/gen32では、unroll基準 `35.314 tok/s` から `35.235 tok/s` で誤差程度。RDNA2は従来どおり `rows_per_block=1` のままなので、ほぼ中立だった。
- R9700 prompt16/gen128では `55.050 tok/s`、skip2 `55.010 tok/s`、last8 `52.855 tok/s`。生成文は通常の説明文で、`verified=true`。
- R9700 controlled v0.3 prompt suiteではmean decode `54.512 tok/s`、min `45.783 tok/s`、max `56.652 tok/s`、mean prefill `47.456 tok/s`、verified all `true`。unroll suite平均 `53.385 tok/s` から `+1.127 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `34.094 tok/s`、min `29.983 tok/s`、max `35.109 tok/s`、mean prefill `37.176 tok/s`、verified all `true`。unroll suite平均 `34.141 tok/s` から `-0.047 tok/s` で中立。
- suite output healthはR9700/V620とも `ok=5`、`warn=1`、`not_evaluated=1`。warnは従来同様 `low_unique_token_ratio`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `9.54e-7`、decode末尾 `6.82e-5` だった。
- `cargo build -p ullm-engine --release`、`cargo test -p ullm-runtime-sys -- --test-threads=1`、`cargo test -p ullm-engine -- --test-threads=1`、`cargo fmt --all --check`、`git diff --check` はpass。

### 次の行動

1. `aq4_matvec_add` 限定のRDNA4 `rows_per_block=8` は、品質を壊さずR9700で小幅に効くため正式採用できる。
2. R9700 suite平均は `54.5 tok/s` まで来たが、既存高速量子化engine級の `~79 tok/s` にはまだ届かない。AQ4フォーマット自体の壁というより、個別kernel tuningの余地がまだ残っている段階だと思います。
3. 次は最新状態でsync timingを取り直し、`aq4_matvec_silu_mul` やqkv/z系のrow groupingを個別に試すか、lm-head/top1の相対支配を先に潰すかを判断する。

## 2026-07-06 AQ4 SiLU-mul RDNA4 row group tuning

### 前回の要点

- `aq4_matvec_add` 限定のRDNA4 `rows_per_block=8` 後、R9700 controlled suite平均は `54.512 tok/s`、V620 controlled suite平均は `34.094 tok/s` だった。
- 最新sync-each-layerではR9700 prompt16/gen32が `54.349 tok/s`、層処理合計が `14.565ms/step`、lm-head/top1が `3.359ms/step` だった。
- component timingでは `mlp_gate_up_activation` がp50 `0.160ms` と最大で、次の局所候補は `aq4_matvec_silu_mul` だった。

### 今回の変更点

- `aq4_matvec_silu_mul` だけRDNA4/gfx12で `rows_per_block=8` を使うようにした。
- `aq4_matvec_add` と `aq4_matvec_silu_mul` のsource側/host launch側でRDNA4 row group override helperを共有し、HIPRTC sourceの `ULLM_AQ4_ROWS_PER_BLOCK` とhost grid計算を一致させた。
- 汎用 `aq4_matvec`、pair/triple、qkv/z/gate-beta、gate-beta単独のrow groupingは変更していない。global `rows_per_block=8` は過去に生成崩壊したため、引き続き不採用。
- R9700 prompt16/gen32では、matvec-add rpb8基準 `56.950 tok/s` から `58.465 tok/s` へ改善。skip2は `58.419 tok/s`、last8は `58.012 tok/s`。
- V620 prompt16/gen32では、matvec-add rpb8基準 `35.235 tok/s` から `35.289 tok/s` で中立。RDNA2は従来どおり `rows_per_block=1`。
- R9700 prompt16/gen128では `56.399 tok/s`、skip2 `56.358 tok/s`、last8 `53.953 tok/s`。生成文は通常の説明文で、`verified=true`。
- R9700 controlled v0.3 prompt suiteではmean decode `56.101 tok/s`、min `46.838 tok/s`、max `58.277 tok/s`、mean prefill `48.827 tok/s`、verified all `true`。matvec-add rpb8 suite平均 `54.512 tok/s` から `+1.589 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `34.094 tok/s`、min `30.058 tok/s`、max `35.166 tok/s`、mean prefill `36.990 tok/s`、verified all `true`。matvec-add rpb8 suite平均 `34.094 tok/s` と同等。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `1.91e-6`、decode末尾 `6.39e-5` だった。
- 最新sync-each-layerではR9700 prompt16/gen32が `55.444 tok/s`、skip2 `55.416 tok/s`、last8 `55.032 tok/s`。層処理合計は `14.207ms/step`、lm-head/top1は `3.363ms/step`。
- component timingでは `mlp_gate_up_activation` がp50 `0.160ms -> 0.147ms` に低下した。`mlp_down_residual` はp50 `0.104ms`、`qkv_projection` はp50 `0.118ms`、`z_projection` はp50 `0.079ms`、`out_projection_residual` はp50 `0.069ms`。
- `cargo build -p ullm-engine --release`、`cargo test -p ullm-runtime-sys -- --test-threads=1`、`cargo test -p ullm-engine -- --test-threads=1`、`cargo fmt --all --check`、`git diff --check` はpass。

### 次の行動

1. `aq4_matvec_silu_mul` 限定のRDNA4 `rows_per_block=8` は、品質を壊さずR9700で効き、V620で中立なので正式採用できる。
2. R9700 suite平均は `56.1 tok/s` まで上がった。既存高速量子化engine級の `~79 tok/s` にはまだ届かないが、AQ4フォーマット自体の壁ではなく、kernelごとの低遅延化余地がまだ残っている。
3. 次の候補はp50 `0.118ms` の `qkv_projection` と、p50 `0.079ms` の `z_projection`。ただしqkv/z系はglobal rpb8で生成崩壊した可能性がある領域なので、個別kernel・短い直接ベンチ・prompt suite guardの順で慎重に試す。

## 2026-07-06 AQ4 qkv/z/gate-beta RDNA4 row group tuning

### 前回の要点

- `aq4_matvec_silu_mul` 限定のRDNA4 `rows_per_block=8` 後、R9700 controlled suite平均は `56.101 tok/s`、V620 controlled suite平均は `34.094 tok/s` だった。
- R9700 sync-each-layerではprompt16/gen32が `55.444 tok/s`、層処理合計が `14.207ms/step`、lm-head/top1が `3.363ms/step` だった。
- component timingでは `qkv_projection` p50 `0.118ms`、`z_projection` p50 `0.079ms` が残っていた。ただしcomponent timingではqkv/z/gate-beta融合が診断のため無効化されるため、fused kernelの変更効果はsync-each-layer側で見る必要がある。

### 今回の変更点

- R9700/RDNA4通常経路で使う fused `aq4_matvec_qkv_z_gate_beta` だけ、RDNA4/gfx12で `rows_per_block=8` を使うようにした。
- pair/triple、汎用 `aq4_matvec`、単独gate-betaのrow groupingは変更していない。
- global `rows_per_block=8` では以前に生成崩壊したが、今回のfused qkv/z/gate-beta単独変更ではprompt16/gen32、gen128、controlled suiteの生成は壊れなかった。
- R9700 prompt16/gen32では、SiLU-mul rpb8基準 `58.465 tok/s` から `59.027 tok/s` へ改善。skip2は `58.975 tok/s`、last8は `58.523 tok/s`。
- V620 prompt16/gen32では、SiLU-mul rpb8基準 `35.289 tok/s` から `35.215 tok/s` で中立。RDNA2は従来どおり `rows_per_block=1`。
- R9700 prompt16/gen128では `56.932 tok/s`、skip2 `56.890 tok/s`、last8 `54.475 tok/s`。生成文は通常の説明文で、`verified=true`。
- R9700 controlled v0.3 prompt suiteではmean decode `56.547 tok/s`、min `47.149 tok/s`、max `58.719 tok/s`、mean prefill `48.915 tok/s`、verified all `true`。SiLU-mul rpb8 suite平均 `56.101 tok/s` から `+0.446 tok/s`。
- V620 controlled v0.3 prompt suiteではmean decode `34.026 tok/s`、min `30.006 tok/s`、max `34.943 tok/s`、mean prefill `37.527 tok/s`、verified all `true`。SiLU-mul rpb8 suite平均 `34.094 tok/s` から `-0.068 tok/s` でノイズ範囲。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `3.81e-6`、decode末尾 `8.92e-5` だった。
- 最新sync-each-layerではR9700 prompt16/gen32が `56.111 tok/s`、skip2 `56.096 tok/s`、last8 `55.664 tok/s`。層処理合計は `14.001ms/step`、lm-head/top1は `3.363ms/step`。
- component timingは融合を無効化するため今回のfused kernel改善を直接反映しない。参考値としては、非融合componentでは `mlp_gate_up_activation` p50 `0.148ms`、`mlp_down_residual` p50 `0.105ms`、`qkv_projection` p50 `0.119ms`、`z_projection` p50 `0.080ms` だった。
- `cargo build -p ullm-engine --release`、`cargo test -p ullm-runtime-sys -- --test-threads=1`、`cargo test -p ullm-engine -- --test-threads=1`、`cargo fmt --all --check`、`git diff --check` はpass。

### 次の行動

1. fused `aq4_matvec_qkv_z_gate_beta` 限定のRDNA4 `rows_per_block=8` は、小幅ながら品質を壊さずR9700で効いたため正式採用できる。
2. R9700 suite平均は `56.5 tok/s` まで上がったが、既存高速量子化engine級の `~79 tok/s` にはまだ届かない。
3. 次はself-attn側のpair/triple q/k/v/qk、またはpair/triple汎用kernelのrow groupを個別に試す。global rpb8の失敗から、変更単位を広げすぎると生成崩壊する可能性があるため、引き続き1kernel単位で進める。

## 2026-07-06 AQ4 pair/triple RDNA4 row group rejection

### 前回の要点

- fused `aq4_matvec_qkv_z_gate_beta` 限定のRDNA4 `rows_per_block=8` 後、R9700 controlled suite平均は `56.547 tok/s`、V620 controlled suite平均は `34.026 tok/s` だった。
- 最新sync-each-layerではR9700 prompt16/gen32が `56.111 tok/s`、層処理合計が `14.001ms/step`、lm-head/top1が `3.363ms/step` だった。
- 残るAQ4 matvec候補としてself-attn系の `aq4_matvec_triple` と `aq4_matvec_pair` の個別row-group調整を試す価値があった。

### 今回の変更点

- `aq4_matvec_triple` だけRDNA4/gfx12で `rows_per_block=8` にする実験を行った。
- R9700 prompt16/gen32では、qkv/z/gate-beta rpb8基準 `59.027 tok/s` から `59.056 tok/s` でほぼ中立。R9700 prompt16/gen128でも `56.932 -> 56.949 tok/s` でほぼ中立だった。
- R9700 controlled v0.3 prompt suiteではmean decode `56.629 tok/s`、min `47.186 tok/s`、max `58.743 tok/s`、mean prefill `48.508 tok/s`、verified all `true`。前段suite平均 `56.547 tok/s` から `+0.082 tok/s` だが、prefill平均は `48.915 -> 48.508 tok/s` に下がった。
- V620 controlled v0.3 prompt suiteではmean decode `34.055 tok/s`、verified all `true`。前段 `34.026 tok/s` と実質同等だった。
- triple rpb8のcross-device guardはpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `2.86e-6`、decode末尾 `2.62e-5` だった。
- 改善幅がノイズ程度でprefillが悪化したため、`aq4_matvec_triple` rpb8は正式採用しなかった。
- `aq4_matvec_pair` だけRDNA4/gfx12で `rows_per_block=8` にする実験も行った。
- R9700 prompt16/gen32では `59.027 -> 58.913 tok/s`、prompt16/gen128では `56.932 -> 56.852 tok/s` に悪化。V620 prompt16/gen32は `35.215 tok/s` で中立だった。
- pair rpb8はdirect decodeで悪化したため、prompt suiteまでは回さず正式採用しなかった。
- どちらも最終コードには残していない。最新コードはqkv/z/gate-beta rpb8採用後の状態に戻した。

### 次の行動

1. pair/tripleのrow-group調整は、少なくとも現在のrow-paired kernelでは有効な改善源ではない。
2. 残る大きな固定費はlm-head/top1の約 `3.36ms/step` と、層内ではMLP down/residual・qkv/z非融合診断上のprojection・recurrent周辺。
3. 次はAQ4 matvec単体のrow-group調整ではなく、lm-head/top1の削減、または層内kernel launch/fusionの削減を検討する。ただしlm-headはAQフォーマット固有ではないため、AQ dequantの壁を判断するには、先にAQ4 kernel側で大きな未探索余地が残っていないか確認する。

## 2026-07-06 BF16 lm-head paired load

### 前回の要点

- pair/triple row-group調整は不採用にした。採用済みの最新コードは `aq4_matvec_add`、`aq4_matvec_silu_mul`、fused `aq4_matvec_qkv_z_gate_beta` のRDNA4 `rows_per_block=8` まで。
- R9700 controlled suite平均は `56.547 tok/s`、V620 controlled suite平均は `34.026 tok/s` だった。
- R9700 sync-each-layerでは層処理合計が `14.001ms/step`、lm-head/top1が `3.363ms/step` で、lm-headがtoken/s上の大きな固定費として残っていた。

### 今回の変更点

- BF16 matvec kernelで、`cols` が偶数のときに2つのBF16を1つの32bit loadから読むpaired load経路を追加した。
- 奇数 `cols` は従来の16bit load経路へfallbackする。
- これはAQ4固有の改善ではなく、gpu resident lm-headの固定費削減を狙うもの。加算順が変わるため、top1/token/logit guardで品質を確認した。
- R9700 prompt16/gen32では、qkv/z/gate-beta rpb8基準 `59.027 tok/s` から `59.214 tok/s` へ改善。skip2は `59.163 tok/s`、last8は `58.628 tok/s`。
- V620 prompt16/gen32では、qkv/z/gate-beta rpb8基準 `35.215 tok/s` から `35.684 tok/s` へ改善。skip2は `35.661 tok/s`、last8は `35.468 tok/s`。
- R9700 prompt16/gen128は単独再実行で `57.024 tok/s`、skip2 `56.980 tok/s`、last8 `54.551 tok/s`。同時実行で取ってしまった初回gen128/sync結果はGPU干渉があるため無効データとして扱った。
- R9700 sync-each-layer単独再実行では `55.888 tok/s`。lm-head/top1平均は `3.363ms -> 3.320ms` に下がったが、層処理側の揺れで全体syncは前段より低く出た。
- R9700 controlled v0.3 prompt suiteではmean decode `56.683 tok/s`、min `46.734 tok/s`、max `59.034 tok/s`、mean prefill `48.007 tok/s`、verified all `true`。qkv/z/gate-beta rpb8 suite平均 `56.547 tok/s` から `+0.136 tok/s`。prefill平均は `48.915 -> 48.007 tok/s` で悪化方向だが、直接計測ではlm-head単体は短縮しており、層側の揺れを含む可能性が高い。
- V620 controlled v0.3 prompt suiteではmean decode `34.578 tok/s`、min `30.410 tok/s`、max `35.582 tok/s`、mean prefill `37.373 tok/s`、verified all `true`。qkv/z/gate-beta rpb8 suite平均 `34.026 tok/s` から `+0.551 tok/s`。
- R9700/RDNA4とV620/RDNA2のcross-device prompt suite guardは `atol=1e-3` でpass。7/7 caseで生成tokenとtop logitsが一致し、最大logit差はprefill `2.86e-6`、decode末尾 `8.87e-5` だった。
- `cargo build -p ullm-engine --release`、`cargo test -p ullm-runtime-sys -- --test-threads=1`、`cargo test -p ullm-engine -- --test-threads=1`、`cargo fmt --all --check`、`git diff --check` はpass。

### 次の行動

1. BF16 lm-head paired loadは、R9700で小幅、V620で明確にdecode平均を上げ、品質guardも通ったため正式採用できる。
2. ただしこれはAQ4フォーマット固有の問題ではない。AQ4側はrow-group・paired index load・固定長unroll・one-pass化まで進み、局所的な未探索余地は小さくなってきた。
3. R9700 suite平均は `56.7 tok/s` で、既存高速量子化engine級の `~79 tok/s` にはまだ届かない。ここから先はAQ4 dequant単体より、層内のkernel launch/fusion、linear-attn recurrent/post、またはlm-head/top1の構造的な統合が必要になる可能性が高い。

## 2026-07-06 BF16 lm-head multi-row block rejection

### 前回の要点

- BF16 lm-head paired load後、R9700 controlled suite平均は `56.683 tok/s`、V620 controlled suite平均は `34.578 tok/s` だった。
- paired loadでR9700のlm-head/top1平均は `3.363ms -> 3.320ms` に下がったが、token/s全体の改善は小さかった。
- 次のlm-head候補として、1 block 1 rowではなく、1 block 4 rowsにして64 threads/rowを維持するmulti-row blockを試した。

### 今回の変更点

- BF16 matvec kernelを実験的に `rows_per_block=4`、`threads_per_row=64`、block size 256に変更した。
- R9700 prompt16/gen32では、paired load基準 `59.214 tok/s` から `59.071 tok/s` に悪化した。
- V620 prompt16/gen32では、paired load基準 `35.684 tok/s` から `35.859 tok/s` に改善した。
- R9700側で悪化し、V620側の改善幅も小さいため、RDNA別分岐を追加するほどの価値はないと判断した。
- multi-row block実験は正式採用せず、コードはBF16 paired load採用状態へ戻した。

### 次の行動

1. lm-headの単純なblock形状調整は、少なくともR9700では有効ではない。
2. これ以上lm-headで大きく伸ばすには、matvecとtop1の構造的な統合や、より根本的なvocab projection戦略が必要になる。
3. AQ4 dequant局所最適化と単純なlm-head調整は、かなり収穫逓減に入った。次に大きく伸ばすなら、層内kernel fusionまたはdecode loop全体のlaunch削減が主戦場になる。
