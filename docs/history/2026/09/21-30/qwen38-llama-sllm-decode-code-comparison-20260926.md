# Qwen3.8 decode：llama.cppとsLLMのコード比較（2026-09-26）

## 対象と判断範囲

ユーザー依頼により、[実行時間内訳](qwen38-q5kxl-llama-and-sllm-long-prefill-20260926.md#追補r9700-hipの定常decode時間内訳)で差が大きかった行列積、Attention、GDN＋convを直接比較した。R9700 `gfx1201`、MTPなし、文脈約8192、最後の8定常Graph replayを対象とする。

- llama.cpp: 固定`fcc891545b0f06de346d8f67d1e6c61f9bf0e777`、Q5_K_XL GGUF、F16 KV。
- sLLM: profile時binary `1cdc2452cbade7cc2d2f7344cb5739eb6d8eec54cc7e4cc9625baa5f3d5a6575`、NVFP4/FP8混合weight、MXFP8 E4 KV、固定GPU sampling。
- source、保存済みprofile、model metadata、既存code object/ISAの読み取りだけを行った。追加GPU計測、production変更、外部sourceの新規copy/adapt/portは行っていない。
- 下記の時間はprofiler下のkernel時間である。形式・丸め・融合範囲が違うため、同一数値契約のkernel A/B、通常TPOTの寄与、採用可能な短縮量を示すものではない。

## 結論の要約

| 項目 | 確認した主な構造差 | 原因としての解釈 |
| --- | --- | --- |
| 行列積 | 両者497行列・約19 GBのweight。llama.cppは64組のgate/upを融合して433 matvec。sLLMは497 matmul。NVFP4はblock16 scale、4出力/wave、VGPR96。llama.cppは1出力/WGを8 waveでK方向へ分担 | weight量の差だけでは約4 ms差を説明できない。融合、dot/scale方式、K方向並列化・register負担が候補 |
| Attention | llama.cppは2 Q headでKV共有、32 key単位のsoftmax更新、packed half2。sLLM hot pathは1 Q head/wave、key単位のsoftmax、MXFP8展開 | 量子化KVのbyte削減がQ head間の再利用不足で相殺される。逐次softmaxと展開の費用が有力候補 |
| GDN＋conv | llama.cppは1 headを32 WGへ分割しstateをregister保持。sLLMは1 head/1 WG、stateを2回走査し、norm/gate等も内部融合 | 並列度・アクセス・state走査・数値契約に差。ただし融合境界を揃えるとkernel時間差は約0.65 msで、前回の約1.64 msより小さい |

個別変更が何msを生んだかを切り分けるablationは行っていない。構造差が存在することと、その性能寄与が測定済みであることを区別する。

## 1. 行列積：同じ読量でもdotと分割方式が異なる

### 実行対象とweight量

実weight metadataを集計した。embedding全体、未使用MTP、conv、normを除き、投影／MLP／lm_headと保存scaleを含む。

| 項目 | llama.cpp | sLLM |
| --- | ---: | ---: |
| 論理weight行列数 | 497 | 497 |
| 行列weight payload | 18.984 GB | 19.098 GB |
| 行列積kernel起動数 | 433 | 497 |
| profile kernel時間 | 32.006 ms | 36.048 ms |
| payload÷kernel時間 | 593.1 GB/s | 529.8 GB/s |

weight payload差は約0.6%で、単にllama.cppが小さいモデルを読んでいるという説明にはならない。換算帯域は有効payload/timeであり、cache・重複loadを含む物理VRAM counterではない。

sLLMの内訳はFP8 10.629 GB／18.490 ms＝574.8 GB/s、NVFP4 8.423 GB／17.194 ms＝489.9 GB/s、小さいBF16投影0.047 GB／0.363 ms＝129.9 GB/s。FP8側に比べ、NVFP4側のpayload換算効率が低い。これはshape、融合、codecも含む観測であり、単一形式だけの速度比ではない。

### gate/up融合

llama.cppの[mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/fcc891545b0f06de346d8f67d1e6c61f9bf0e777/ggml/src/ggml-cuda/mmvq.cu#L741)では、Q5_Kの通常積とgate積を同じK loopで計算し、reduction、SiLU、gate×upまで同kernel内で実行する。今回のQ5_K行列272個は、非融合144＋融合64の計208 matvecになり、64組のMLP gate/upが497→433の差を説明する。

sLLMの[qwen38_projection_pack_runtime.inc](../../../../../native/hip/src/qwen38_projection_pack_runtime.inc)の1535行以降は、activation量子化を共有した後、2 memberを順に`lowp_matmul_launch`へ渡す。gate/upの積本体は別kernelで、SiLU×upもproducer kernel側で行う。

llama.cppの融合は活性値の利用・reduction・dispatchを共有できる。ただし、llama.cppのF32中間とsLLMのBF16中間ではbyte幅が異なるため、buffer数だけから中間trafficの削減byte数を推定しない。約4.0 msのkernel時間差を融合だけへ帰属しない。

### arithmeticと出力タイル

llama.cpp Q5_Kは256要素super-block（176 B）、32要素単位のscale/min補正、Q8_1 activationとinteger DP4Aを使う。sLLM NVFP4は16要素blockのE2M1 weight/activationをsigned byteへ展開し、signed dot4とweight/activationのscaleをblock単位でFP32累積へ反映する。sLLM FP8はnative FP8 dot4を使い、row scaleを出力時に適用する。4bit保存であることだけからNVFP4の演算負担が小さいとは言えない。

今回の実行構成（profilerのGridはglobal workitem数としてWGサイズで割った）:

| 経路 | WG内の配置 | 主なVGPR | LDS | 役割 |
| --- | --- | ---: | ---: | --- |
| llama Q5_K fused | 32×8 thread、1出力/WG | 48 | 2048 B | 8 waveで同じ出力のK reductionを分担 |
| llama Q6_K fused | 32×8 thread、1出力/WG | 32 | 2048 B | 同上 |
| sLLM NVFP4 actshared | 256 thread、32出力/WG | 96 | 1536 B | 1 waveが4出力を保持しactivationを共有 |
| sLLM FP8 dot4 | 256 thread、8出力/WG | 32 | 0 | 1 waveが1出力を計算 |

例えばN17408のQ5_K fusedは17408 WG、sLLM NVFP4は544 WG。llama.cppはK方向へ多くのthreadを投入する一方、sLLMは出力方向でactivationを再利用する。NVFP4のVGPR96はregisterによる同時resident wave制約の候補だが、実occupancy counterはこの比較で取得していない。traceのScratchは0で、spillを差の根拠にはしない。

参照: llama.cpp [calc_rows_per_blockとrow mapping](https://github.com/ggml-org/llama.cpp/blob/fcc891545b0f06de346d8f67d1e6c61f9bf0e777/ggml/src/ggml-cuda/mmvq.cu#L579)、sLLM [NVFP4 shared body](../../../../../native/lowp/src/nvfp4_decode_scale_lut.inc)、[FP8 dot4](../../../../../native/hip/src/matmul_kernel.hip.cpp)。

### 除外した原因：unused activation prefetch

NVFP4のprefetch helperには、shared bodyで使わないactivation fieldsがソース上残る。しかし既存の保存済みnative ISAでは、そのglobal loadはDCEされていた。activation stagingのpack/scale load以降はweight/scale loadだけで、余分なactivation再読出しを性能原因に数えない。

profile時の`1cdc...` executable自体は、その後共有作業treeで置き換わっていた。保存済み前段binary `0fd0ce0d...`のgfx1201 code objectと現在binaryのobjectは同一SHA-256 `904e216683c814004a46da02d7836f5b1d0624dd43af6a6b9a2b92d0c1d367c4`で、actshared ISAも一致した。`0fd...`→`1cdc...`のbuild logはRust `sllm-hip`だけの再compileを記録しており、native HIP再compileはない。この記録を対応付けの根拠とし、現在の異なるexecutable hashをprofile時hashへ読み替えない。

## 2. Attention：GQA共有とsoftmaxの処理単位

### KVの再利用

Qwen3.8はQ head 24、KV head 4、head dim 256、GQA比6。llama.cppの実行kernelは`flash_attn_tile<256,256,1,2,false>`で、同じblockが2 Q headを担当し、LDSへ読んだK/Vを両headで使う。

sLLMのgfx1201 hot pathは1 waveが1 Q headと1 splitを担当し、同じKV headを6つのQ headが別々に読み、展開する。[causal_attention_kernel.hip.cpp](../../../../../native/hip/src/causal_attention_kernel.hip.cpp)の1435行以降にhead/split mapping、1511行以降にpage loopとkey loopがある。

8192 token／1層で、cacheを考えない論理要求量の目安:

| 経路 | 一意のK/V payload | head共有後の要求倍率 | 要求量の目安 |
| --- | ---: | ---: | ---: |
| llama.cpp F16 KV | 33.554 MB | 3 | 100.663 MB |
| sLLM MXFP8 E4 KV | 17.302 MB | 6 | 103.809 MB |

sLLMの小さいKV形式による利点は、Q head間の共有がないことでほぼ相殺される。この表は物理DRAM trafficの測定ではない。

### online softmaxとpacked算術

llama.cppの実行設定は2 warp、32 key単位のFA tile、K側64要素tile。32 keyのscoreをまとめてからmax reduction、softmax状態の再スケール、expの合計を行う。sLLMはsplit内のkeyを1件ずつ処理し、各keyでwave dot reduction、max、`expf` 2回、denominator更新、rescale/contribution broadcast、V累積の再スケールを行う。オンライン状態の更新回数は大きく異なるが、time差を32倍の演算差と読み替えない。

llama.cppは`FAST_FP16_AVAILABLE`のHIP pathでQ/K/Vをhalf2に保持し、KQにはpacked FP16 dotを使う。KQ tileとVKQ accumulatorにもFP16/half2を使う。sLLMはMXFP8 K/Vをhot loopで展開し、FP32でscore／online softmax／value累積を行う。gfx1201の該当providerはE4 decodeとE8M0 scale decode/乗算が分かれたpathである。

したがって、GQA共有不足、keyごとの展開とsoftmax更新、packed演算・precisionの差が複合した候補となる。packed FP16をそのまま採れば同じ数値契約になるわけではない。参照: llama.cpp [head mapping／shared tile](https://github.com/ggml-org/llama.cpp/blob/fcc891545b0f06de346d8f67d1e6c61f9bf0e777/ggml/src/ggml-cuda/fattn-tile.cuh#L851)、[tile softmax](https://github.com/ggml-org/llama.cpp/blob/fcc891545b0f06de346d8f67d1e6c61f9bf0e777/ggml/src/ggml-cuda/fattn-tile.cuh#L606)。

### 小さい補助費用

8回平均の主stage1はllama.cpp 0.953 ms、sLLM 3.929 ms。combine／mergeは約0.061／0.164 ms。sLLMの非選択P32 graph nodeはdevice controlで即returnし、約0.019 msにとどまる。split切替用nodeの起動やmergeだけでは3 ms超の差を説明できない。page lookupも追加処理ではあるが、この資料は差をpaging単独へ帰属しない。

## 3. GDN：融合と並列度のトレードオフ

### 並列配置とstate memory

両者のrecurrent stateはFP32。llama.cppは128×128状態を4列ずつ1 WGへ割り当て、1 headあたり32 WG、48 headで1536 WGとする。1 warpが1列、各laneは4要素をregister保持し、更新後に一度だけ書く。stateは転置配置でwave内laneのload/storeが連続する。

sLLM generic kernelは1 value headを128-thread WGで処理するため48 WGのみ。各threadが1 output dimensionを持ち、128 key dimensionを逐次走査する。gfx1201ではrow配置なので、各thread内は連続だが、同一命令のwave内laneはstride-128でstateを触る。decay＋projection passでstateをread/writeし、その後rank-one update＋output projection passで同stateを再度read/writeする。

参照: llama.cpp [gated_delta_net.cu](https://github.com/ggml-org/llama.cpp/blob/fcc891545b0f06de346d8f67d1e6c61f9bf0e777/ggml/src/ggml-cuda/gated_delta_net.cu#L31)、sLLM [layout helper](../../../../../native/hip/src/linear_attention_kernel.hip.cpp)の70行付近とgeneric kernelの198〜365行。

llama.cppは1536 WG、VGPR32、LDS0。sLLMは48 WG、VGPR56、LDS2048 B。少ないWGによるGPU underfill、stateの再走査、wave内の非連続accessが有力な構造差である。layoutだけの変更は過去のR9700で退行したため（[Phase 9履歴](../../08/11-20/phase9-engine-structural-optimization.md)）、転置だけで改善すると断定しない。work分割・数値順序・fusionと組み合わせて考える必要がある。

### 数値契約と融合範囲

sLLMはQ/K正規化、betaのBF16丸め、decay計算、出力のBF16丸め、RMSNorm、z-SiLUを同kernel内で処理し、複数のblock barrierを使う。`#pragma clang fp contract(off)`でFMA contractionを禁止している。llama.cppはこれらの多くを別kernelへ分け、GDN本体には同じFMA禁止を置いていない。また、decayをstateへ先に掛けてprojectionするsLLMと、元stateのprojection後にdecayを掛けるllama.cppでは、FP32丸め順も異なる。

1 WGでhead全体を持つsLLMは、output RMSNorm等をblock内で完結できる。一方、llama.cppは列方向へ大きく並列化する代わり、head全体のnorm等を別kernelへ出す。fusionによるlaunch削減と、state処理の並列度がトレードオフになっている。

### 公平な範囲へ時間を集計し直した

前回の0.425対2.061 msはGDN core＋convという名前分類で、llama.cppの外側処理を含まなかった。sourceとtraceのgridからGDN専用の外側処理を選び、同じ意味範囲へ足した。共通のinput/post-MLP norm、投影、Full AttentionのcopyやKV appendは含めない。

| llama.cppのGDN関連処理 | ms/token |
| --- | ---: |
| recurrent core＋conv/SiLU | 0.425 |
| Q/K L2 norm | 0.149 |
| alpha softplus/A、beta sigmoid | 0.134 |
| 出力RMSNorm、z gate | 0.139 |
| recurrent state gather | 0.307 |
| conv history gather／concat／copy | 0.260 |
| **kernel時間合計** | **1.414** |

sLLM側の対応するfused recurrent＋convは2.061 ms。範囲を揃えたkernel時間差は約0.647 ms、比は約1.46倍である。core単独の5.6倍差や、旧category差1.636 msをそのまま全体改善余地と扱わない。

llama.cppの外側GDN専用kernelは、`k_get_rows_float`のgrid `(256,120,1)`がconv state（48回）、`k_get_rows_float_vec`がrecurrent state（48回）、`concat_cont`がconv window（48回）、`cpy_scalar`のgrid `(30720,1,1)`がconv history（48回）。embeddingのgatherとFull Attentionのcopyはgridで除いた。出力RMSNormはgrid `(12288,1,1)`の48回、Q/K L2 normは96回。source側の[conv state構築](https://github.com/ggml-org/llama.cpp/blob/fcc891545b0f06de346d8f67d1e6c61f9bf0e777/src/models/delta-net-base.cpp#L449)、[recurrent state gather](https://github.com/ggml-org/llama.cpp/blob/fcc891545b0f06de346d8f67d1e6c61f9bf0e777/src/llama-graph.cpp#L3487)、[Qwen GDN graph](https://github.com/ggml-org/llama.cpp/blob/fcc891545b0f06de346d8f67d1e6c61f9bf0e777/src/models/qwen35.cpp#L340)と対応した。

これはkernel時間の合計であり、pipelineごとのlaunch gapを分離したwall時間ではない。llama.cppの関連kernelは576起動、sLLMは96起動であり、sLLMのfusionによるgap削減も評価に残す必要がある。conv単体にもF32対BF16の入力／weight／roundingとhistory更新の違いがある。

## 証跡

数値正規化とGDN再集計はignored `.local-artifacts/benchmark-20260926/decode-breakdown/code-comparison-normalization.json`、解析入口は`refine-decode-causes.py`。最後の8 graphすべてでGDN関連576 dispatch（48回×10種類＋Q/K norm96回）を照合し、未分類の追加GPU計測は行っていない。初期profileとsource観察だけで原因候補を整理した資料であり、実装変更・採用判定の完了記録ではない。

先行: [速度／profile記録](qwen38-q5kxl-llama-and-sllm-long-prefill-20260926.md)。
