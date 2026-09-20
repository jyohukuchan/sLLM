# Phase 87 段階0: W×A16 棚卸し

## 結論

現行の本番 Qwen3.8-27B Unsloth 経路は、段階0着手時の「M=1 の NVFP4 は W4A16」という前提と一致しない。直接artifactの graph binding は `Encoding::Nvfp4W4A4` であり、HIP runtime も `metadata.nvfp4_w4a4` を `MatmulFormat::Nvfp4W4A4` へ送る。したがって、Qwen3.8本体については W4A16 から W4A4 へ移行する未実施作業は現在のsource上は存在せず、現行W4A4の速度・KLDを基準として確認する作業が残る。

W4A16は、低精度ライブラリの公開ABIと、旧来の verified NVFP4 sidecar を使う Qwen3.5／Gemma 4 の互換経路には残っている。MXFP8 W8A16／MXFP6 W6A16 は Phase 85 の M=1 opt-in（ID101／102）であり、Qwen3.8 MTP用の MX sidecar evidence が主な実利用記録である。

Qwen3.8の段階0 baseline実行でもこの区別を確認した。`.local-artifacts/phase87/stage0/baseline-gfx1201/baseline-mtp-off/report.json` は exact `gfx1201`、MTP無効、1 warmup＋3 measured、全dispatch HIP、fallback falseで、NVFP4 decodeは kernel ID84 `matmul.nvfp4.w4a4.decode.scale_lut.v1` が各run 14,224 dispatch、prefillはID89 `matmul.nvfp4.w4a4.prefill.gfx1201.wmma128x64.kahan.v1` が448 dispatchだった。これは現行Qwen3.8本体の実行形式がW4A4である直接のruntime evidenceである。

## モデルとartifact

| 対象 | 現在の経路 | W4A4に必要な入力global scale | 判定 |
| --- | --- | ---: | --- |
| `unsloth/Qwen3.8-27B-NVFP4`、本体 | `/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4/model.safetensors`（22,568,192,096 bytes、header 251,128 bytes、header SHA-256 `4b84e021…22f9e4`）。56層×MLP gate/up/down の168 tensorが `Nvfp4W4A4`。attention/GDNと最終MLPはFP8、残りはBF16 | **168/168**。headerは全1,953 tensor、`input_global_scale` 168、`weight_global_scale` 168、FP8 channel `weight_scale`を含む | W4A4対応済み。W4A16本体ではない |
| Qwen3.8 MTP companion | 本体とは別の MTP sidecar。通常の既定はBF16、Phase 85 evidenceでは MXFP8 W8A8／MXFP6 W6A6、M=1だけW8A16／W6A16 opt-in | 不要。W8A8／W6A6は動的activation scale、W×A16撤去後の置換先は同じW8A8／W6A6 | ID101／102を削除対象として保持 |
| `unsloth/gemma-4-12b-it-NVFP4`、第一級artifact | `/home/homelab1/.cache/sllm/models/unsloth--gemma-4-12b-it-NVFP4/model.safetensors`（9,304,966,064 bytes、header 179,720 bytes、file SHA-256 `7c2ee232…ce704b`、header SHA-256 `23a75ce4…3bd5d55`）。48層×3 MLP = 144 tensorを `Encoding::Nvfp4W4A4` としてresident化。attention 184 tensorはFP8、static KV scaleは48層 | **144/144**。header全1,389 tensor中、`input_global_scale` 144、`weight_global_scale` 144、`k_scale`／`v_scale`各48。GGUF派生物もNVFP4 type 144 tensor、recipe input scale binding 144件 | W4A4対応済み。W4A16は第一級artifactの実行形式ではない |
| Gemma 4 12B の旧NVFP4 sidecar | `Gemma4ResidentModel::new_nvfp4` → `nvfp4_tensor_view` → `Encoding::Nvfp4`。BF16 activationを直接読むW4A16 | **なし**。sidecarはvalue、block scale、weight tensor scaleだけを保持し、converter metadataも `input_global_scales_applied=false`。ただし保存済みPhase 15Q evidenceはS0 variant `tensor_count=144` と artifact SHA-256 `bf03e10f…63884f` を記録する | 旧evidence／互換経路。W4A4移行は今回のQwen3.8本体比較対象ではない |
| `nvidia/Gemma-4-26B-A4B-NVFP4` | ローカルHF cacheのrevision `a19cfe00…fdcffe6`。indexは47,033 tensor、2 shard、合計18,782,360,732 bytes。shard headerは1号2,806,200 bytes／tensor 21,603、2号3,318,640 bytes／tensor 25,430。custom layer blobは30層×128 expert×gate/up/down = **11,520** | **11,520/11,520**。indexは`input_scale` 11,520、`weight_scale`系23,040（うち`weight_scale_2` 11,520）を持ち、shard headerのF32/U8/F8_E4M3構成とも一致 | custom W4A4相当。generic `LOWP_NVFP4_W4A16` consumerではない |
| `nvidia/Gemma-4-31B-IT-NVFP4` | ローカルpayloadはない。locked revision `4135a98a…82223ac` のHF config／quant config／indexをremote readした | **180/180**。upstream indexは1,728 tensor、合計32,633,255,032 bytes、`input_scale` 180、`weight_scale` 360。`hf_quant_config.json` は `quant_algo=NVFP4`、`group_size=16`、KV=`FP8` | source上はscaleあり。ただしlockは`schema-and-reference-only`、runtime constraintは32 GiB single GPU超過であり、実行対象ではない |

Qwen3.8の直接loaderは `crates/sllm-core/src/qwen_graph.rs` の `build_qwen35_unsloth_qwen38_nvfp4_graph` で NVFP4を `Encoding::Nvfp4W4A4` に固定している。入力scaleは `crates/sllm-core/src/quantized_model.rs` の `read_qwen38_nvfp4_input_global_scale_f32_bits` が56層×3 projectionを走査し、168件でfail-closedする。第一級Gemma artifactも `crates/sllm-core/src/gemma4_execution.rs` の `quantized_gemma_tensor_view` で同じ `Encoding::Nvfp4W4A4` に変換され、lock inventoryはMLP 144件を固定する。

26B MoEは通常のlowp matmulではなく、`native/hip/src/moe_expert_kernel.hip.cpp` の `sllm_gemma4_moe_quantize_active_nvfp4_v2`、`sllm_gemma4_moe_gateup_nvfp4_v2`、`sllm_gemma4_moe_down_combine_nvfp4_v2` を使う。custom blobの `kGateInputScalesOffset`、`kUpInputScalesOffset`、`kDownInputScalesOffset` は各128 expert分のF32 scalarを持つ。source側の件数契約は `GEMMA4_MOE_EXPERT_PROJECTION_COUNT = 30 * 128 * 3` である。

ローカル実体でも次を確認した。Gemma 4 12Bの派生GGUF `/home/homelab1/.cache/sllm/derived/phase20-final-gemma4-nvfp4.gguf` は9,337,229,760 bytes、GGUF tensor 1,149件、NVFP4 tensor type 40が144件、scale tensorが472件（NVFP4 input/outer各144とFP8 channel 184）で、lockのfile SHA-256は `sha256:4e041c6a…e902fb5`、metadata SHA-256は `sha256:b8d7c83b…54212e`、tensor catalog SHA-256は `sha256:b3c6f97d…7924d4` である。第一級safetensors headerと派生GGUF recipeの両方で144 input scaleを確認できる。

26B MoEは `/home/homelab1/.cache/huggingface/hub/models--nvidia--Gemma-4-26B-A4B-NVFP4/snapshots/a19cfe00be84568a6867111c9a68c9c44fdcffe6/model.safetensors.index.json` と2 shard headerを実読した。1号shardはfile SHA-256 `b5df3112…6d819`、2号は `ff11061e…10962` で、各headerのF32／U8／F8_E4M3 plane数を合算すると、indexの11,520 input scale、23,040 block/secondary scaleと一致する。

31Bはローカルにpayloadがないため、locked revisionの[HF config](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4/blob/4135a98a9b728a548947683219633b25682223ac/config.json)、[quant config](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4/blob/4135a98a9b728a548947683219633b25682223ac/hf_quant_config.json)、[safetensors index](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4/blob/4135a98a9b728a548947683219633b25682223ac/model.safetensors.index.json)をrange readした。index上のinput scale 180件は確認できるが、shard payload header／実行runtimeは確認していない。

## W4A16 runtime／provider／selector

### 公開ABIとlowp provider

`native/lowp/include/lowp/lowp.h` は次を公開値として残している。

| ABI | 意味 |
| ---: | --- |
| `2` | `LOWP_NVFP4_W4A16` |
| `6` | `LOWP_MXFP8_E4M3_W8A16` |
| `7` | `LOWP_MXFP6_E3M2_W6A16` |

`native/lowp/include/lowp/detail/low_precision_matmul_provider.hpp` の `MatmulFormat`、`ProviderKind::Nvfp4W4A16Block16`、`ProviderKind::Mxfp8A16Block32`、`ProviderKind::Mxfp6A16Block32` が対応する provider identityである。`native/lowp/src/lowp_plan.cpp` は W8A16／W6A16を M=1 専用 variantへ送り、W4A16は `select_nvfp4_variant` を使う。`native/lowp/src/lowp_launch.cpp` は BF16 activationとweight tensor scaleを受けて `launch_nvfp4` を呼び、W8A16／W6A16はそれぞれ `launch_mxfp8_w8a16`／`launch_mxfp6_w6a16` を呼ぶ。

HIP public runtimeの入口は `native/hip/src/matmul_runtime.inc::matmul_lowp_format` である。`metadata.nvfp4_w4a4` を先に検査するため、Qwen3.8／第一級GemmaのW4A4 descriptorはW4A4へ進む。W4A16へ進むのは旧 `SLLM_HIP_MATMUL_NVFP4_VERSION` descriptor（`metadata.nvfp4` かつ `nvfp4_w4a4=false`）だけである。MX A16は `M=1 && SLLM_MX_WA_M1_A16=1 && !SLLM_MX_WA_M1_FORCE_BASELINE` のときだけ選ばれる。

GGUF converter／runtimeもこの区別を維持する。`crates/sllm-core/src/gguf_convert.rs::build_gemma4_nvfp4_gguf_plan` はNVFP4 value、block scale、weight outer scale、input scaleを別tensorとして出力し、`Gemma4ResidentModel::new_gguf_quantized` → `quantized_gemma_tensor_view` は `Encoding::Nvfp4W4A4` を使う。Qwenの `build_qwen35_gguf_mixed_graph` と直接Qwen3.8 graphも、recipeのInput scale planeがあるNVFP4 bindingを `Encoding::Nvfp4W4A4` にする。旧sidecarだけが `Encoding::Nvfp4` と旧W4A16 providerを使う。

### Kernel symbol／selector ID

- NVFP4 W4A16: `sllm_matmul_nvfp4_block16_packed_dequant_v1`（decode）、`sllm_matmul_nvfp4_block16_prefill_row8_tiled256_v2`（prefill）。
- MXFP8 W8A16: logical ID `101`、`matmul.mxfp8.w8a16.m1.col2.v1`、device symbol `sllm_mxfp8_w8a16_m1_col2_v1`。
- MXFP6 W6A16: logical ID `102`、`matmul.mxfp6.w6a16.m1.col2.v1`、device symbol `sllm_mxfp6_w6a16_m1_col2_v1`。

W8A16／W6A16のselectorと環境変数は `native/lowp/include/lowp/detail/lowp_kernel_internal.hpp`、`native/hip/src/matmul_runtime.inc`、`crates/sllm-hip/src/bin/sllm-mxfp-wa-evidence.rs` に分散している。Phase 87段階4で削除する場合は、公開enum、provider、selector、kernel symbol、ABI validation、evidence tool、fixtureを同じ変更単位で扱う必要がある。

## evidenceとfixture

### W4A16

- `ci/tools/lowp_baseline_probe.cpp` は `mxfp8_w8a16`、`mxfp6_w6a16`、`nvfp4_w4a16` をbaseline形状へ列挙する。
- `native/lowp/tests/selection_fixture.csv` と `ci/matrix/lowp-selection-baseline-v1.json` は3形式の境界・shape selector rowsを保持する。これはhost selector evidenceであり、Phase 87段階0のGPU PASSではない。
- `crates/sllm-hip/src/bin/sllm-nvfp4-matmul-evidence.rs` は合成BF16 activationを直接渡すPhase 15 W4A16 operator evidenceで、W4A16の入力global scaleを要求しない。`crates/sllm-hip/src/bin/sllm-nvfp4-qwen-accuracy-evidence.rs` は旧Qwen sidecar、`crates/sllm-hip/src/bin/sllm-phase15q-gemma-accuracy.rs` は旧Gemma sidecarを検証する。後者の保存済み `/home/homelab1/.cache/sllm/evidence/phase15q/analysis.json` はquantized source tensor 1,389件、派生sidecar variant 144件を記録し、`operator-gfx1030.json`／`operator-gfx1201.json` は各7 shape、fallback false、cleanup 0である。これらは第一級Qwen3.8／Gemma12B direct artifactのW4A4 runtime evidenceではない。

### MX A16（ID101／102）

`crates/sllm-hip/src/bin/sllm-mxfp-wa-evidence.rs` に `SLLM_MX_WA_M1_A16`、ID101／102、kernel／device symbolの固定値がある。公開履歴 [Phase 85 A16](phase85-m1-a16-mtp.md) と集約JSON [ci/matrix/phase85-a16-mtp-results-v1.json](../../../../../ci/matrix/phase85-a16-mtp-results-v1.json) の記録では、M=1の実6形状＋K=2016/2048/2080/17376、N=1023/1024/1025境界を両GPUで検証し、operator比較は72件、MTPは8192/128、priming 1024／2048、1 warmup＋3 measuredで比較している。Phase 85の結論はBF16既定維持、A16は明示opt-in維持である。

MX A16はBF16 activationをそのまま読む形式なので、NVFP4 W4A4のようなartifact由来のactivation global scaleを持つ必要はない。W8A16／W6A16の置換先W8A8／W6A6では、activation scaleはruntime quantizerが各M/K blockから動的に生成する。

## 未確認事項と段階1以降への境界

1. 現在のworkspaceには、legacy Qwen／Gemma sidecarの全manifest・artifact payloadは揃っていない。sidecar schemaとconverter contract、および保存済みPhase 15Q evidenceから、旧W4A16 laneの実体件数と「input global scaleを保持せずW4A16で適用しない」契約は確認できるが、旧Qwen sidecar各配布物の実ファイルsha256は未確認である。
2. Qwen3.8 direct artifact（168件）、Gemma4 12B direct artifact（144件）、Gemma4 26B MoE（11,520件）、upstream参照Gemma4 31B（180件）はW4A4に必要な入力scaleをartifact/index上持つ。現行Qwen3.8の基準はW4A4であり、W4A16との移行速度／KLD比較は行わない。
3. Gemma4 31B NVIDIAはscale catalogだけをupstream indexで確認したreference-only対象で、W4A4移行済みやW4A16削除後の動作を主張しない。
4. この棚卸しで新規GPUは実行していない。既存Qwen3.8 W4A4 baselineとPhase 15Q W4A16 evidenceは形式・実行経路の同定にだけ使い、今回の訂正後スコープではW4A16診断比較を追加しない。

計画: [Phase 87](../../../../plans/active/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)
