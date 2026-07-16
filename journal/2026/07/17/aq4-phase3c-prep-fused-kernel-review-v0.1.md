# AQ4 Phase 3c-prep fused kernel review v0.1

## 前回の要点

- Phase 3b（commit `4d04ff1d`）で、07/14の最終相対L2 `0.6151289249` はGateway実requestではなく、production packageを直接loadしたM=1/cold診断の結果と確認した。最初のdecoder layer出力から差があり、warm state、M>1、RoPE、paged KVによる説明（H6）は棄却された。
- CPU参照は個別AQ4 matvecとhost recurrentを使う一方、production M=1はQKV/Z/A/B/gate/beta融合、device recurrent、AQ4 `matvec_add`、fused MLPを使う。従ってH5（GPU kernel実装差）が最優先の未検証仮説である。
- 07/15の限定済みfused probeは、QKV/Z/gate/betaについてCPUとの差が概ね相対L2 `1e-6`以下だった。ただしfull production M=1 layerの証明ではないため、今回の静的レビューで代数・index・scale契約を再確認した。

## 今回の変更点

### 実施範囲

- GPUを列挙・初期化・実行していない。production service、systemd unit、active manifest、07/16停止中P3 harnessにも変更を加えていない。
- `runtime/src/ullm_runtime_hiprtc_sources.inc` のgeneric AQ4 dequant、QKV/Z/A/B/gate/beta fused kernel、fused MLP、`matvec_add`、linear-attention QKV prepare/recurrentを、CPU host実装およびC APIのCPU branch/ABI検証と突き合わせた。
- CPU-only確認として `cargo test -p ullm-runtime-sys cpu_aq4_matvec_ --lib -- --test-threads=1` は10件成功した。これはHIP deviceを列挙・初期化しないCPU test filterである。

### 結論

有効なAQ4 payloadを前提に、07/14のlayer 0からの大差を説明できる**高確信度の通常入力時の算術バグ（欠落項、row scale/bias取り違え、shape/indexずれ、tree reductionのlane欠落）は見つからなかった**。

CPUの要素ごとのdequant/加算とGPUのgroup内raw sum→group scale→tree reductionは丸め順序こそ異なるが、レビューした条件では同じ式になる。従ってH5はまだ棄却せず、Phase 3cで実production M=1のstage値を測る必要がある。

| 範囲 | CPU根拠 | GPU根拠 | 判定 |
| --- | --- | --- | --- |
| AQ4 group scale・row scale | `runtime/src/ullm_runtime_parts/part_00.inc:2701-2735` | `runtime/src/ullm_runtime_hiprtc_sources.inc:623-729` | groupがrow内で完結するfast pathとglobal group fallbackの双方で、codebook×inputのraw sumに一度だけgroup/tensor scaleを掛け、最後にrow scaleを掛ける。丸め順序差のみ。 |
| QKV/Z/A/B/gate/beta fused | `runtime/src/ullm_runtime_api_aq4.inc:2221-2287` | `runtime/src/ullm_runtime_hiprtc_sources.inc:2093-2482`; ABI検証 `runtime/src/ullm_runtime_api_aq4.inc:2033-2348` | QKV/Z各matrixのgroup/row scale、`a + dt_bias`、softplus、`-exp(A_log)*softplus`、B sigmoid、出力rowの対応を確認。欠落項は見つからない。 |
| AQ4 fused MLP | `runtime/src/ullm_runtime_parts/part_00.inc:2871-2934` | `runtime/src/ullm_runtime_hiprtc_sources.inc:2488-2753`; API `runtime/src/ullm_runtime_api_aq4.inc:2488-2557` | gate/upを別scale/row scaleで復号後、`gate * sigmoid(gate) * up`。shapeと引数順も一致。 |
| output projection + residual | `runtime/src/ullm_runtime_parts/part_00.inc:2833-2869` | `runtime/src/ullm_runtime_hiprtc_sources.inc:1250-1386` | AQ4 matvecをrow scaleまで完了してから同じresidualを加算。欠落項なし。 |
| Conv SiLU / Q,K L2 norm | `runtime/src/ullm_runtime_parts/part_01.inc:4822-4881` | `runtime/src/ullm_runtime_hiprtc_sources.inc:5731-5860` | Conv後のSiLU、Q/Kの`+1e-6` L2 norm、Q scaleが双方にある。07/05のConv SiLU欠落と同種の再発は確認されない。 |
| recurrent state | `runtime/src/ullm_runtime_parts/part_01.inc:5801-5856` | `runtime/src/ullm_runtime_hiprtc_sources.inc:6188-6362` | state layout `[value_head,key_dim,value_dim]`、decay→current→beta update→Q readoutの順序が対応。reduction順のみGPUと異なる。 |
| layer input RMSNorm | linear `crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs:5067-5091`、self-attn `:2425-2446` | 同じproduction device operator呼出し | linear/self-attention双方に入力RMSNormがある。07/05のself-attn input RMSNorm未適用と同種の欠落は現行M=1 pathにない。 |

### 見つかった不一致・候補

1. **確実な入力エラー処理差（通常の有効payloadの根因候補としては未確認）**

   - CPU host matvecは `scale_index >= scale_count` をエラーとして止める（`runtime/src/ullm_runtime_parts/part_00.inc:2724-2728`）。通常C APIもCPU bufferについては全scale indexを事前検証する（`runtime/src/ullm_runtime_api_aq4.inc:663-672`）。
   - GPU generic/fused helperは該当groupを黙ってskipしてsumを続行する。genericは `runtime/src/ullm_runtime_hiprtc_sources.inc:657-660`、QKV/Z helperは `:2115-2118,2258-2263`、MLP helperは `:2508-2511,2633-2638`、`matvec_add`は `:1272-1275` にある。
   - fused APIはbuffer長・shapeは検証するが、HIP上のscale-index内容をD2Hして検証しない（`runtime/src/ullm_runtime_api_aq4.inc:2163-2219,2421-2488`）。これはCPUとGPUの明確なsemantic差であり、**不正metadataをsilent corruptionにするバグ**である。
   - ただし07/14 packageに不正scale indexがある証拠は今回得ていない。有効packageなら発火しないため、これを07/14の根因とは断定しない。Phase 4で修正するかは、package metadataの別途検証とPhase 3c結果後に判断する。

2. **確実な条件付きconfiguration/cache bug（07/14根因は未確認）**

   - HIPRTC sourceは初回compile時にRPBを定数として埋め込む。QKV fusionは `runtime/src/ullm_runtime_hiprtc_sources.inc:599-606`、MLPは `:609-615`。module cacheはdevice IDだけをkeyに保つ（QKV `runtime/src/ullm_runtime_parts/part_00.inc:3686-3766`、MLP `:5199-5277`）。
   - 一方launcherは各callでRPB環境変数を再読してgridを決める（QKV `runtime/src/ullm_runtime_parts/part_00.inc:4712-4728`、MLP `:5310-5317`、add `:4320-4327`）。
   - 同一processで初回compile後にRPBを変更すると、kernel内部の`rows_per_block`とlaunch gridが異なる。compile=4→launch=8なら後半row未書込み、逆方向なら範囲外アクセスの可能性がある。これは**条件が満たされた場合は確実な設定cache不整合**である。
   - 07/14にprocess中のRPB変更があった証拠はない。Phase 3cではprocess起動前に全RPB環境を固定し、実行中に変更しないことでこの条件を除外する。

3. **疑わしいが未確認ではなく、丸め順序差として扱う範囲**

   - tree reductionは各levelに`__syncthreads()`があり、RPBは`256 % rows_per_block == 0`に制限される（`runtime/src/ullm_runtime_parts/part_00.inc:68-100`）。reviewed kernelのpartial初期化・lane範囲・barrierに静的な欠落は見つからなかった。
   - CPU逐次加算対GPU tree加算、`std::exp`対`expf`はbit-exactではないが、単独で07/14規模を説明するかはソースだけでは決められない。Phase 3cでfull tensor相対L2を測定する。

### 未解決点

- QKV fused kernelのfull production M=1 pathを直接覆うGPU数値testは存在しない。07/15の限定probeは有益だが、layer stack・recurrent・residual・MLPまで一体で検証していない。
- fusionを保つには `ULLM_SYNC_LINEAR_ATTN_COMPONENTS_FOR_TIMING` を有効にしてはならない。`run_device_step`はこのflagが有効ならQKV/Z/A/B/gate/beta fusionを外す（`crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs:5093-5119`）。Phase 3c traceはこれをfail-closedで拒否する。

## 次の行動

1. GPUを実行せず、layer 0の常駐device bufferを既存D2H経路で読む診断traceを完成・build・unit testする。QKV/Zに加えgate/betaも記録し、fusion内のA/B式とrecurrent以降を分離できるようにする。
2. CPU AQ4 stage streamとGPU stage streamを同じ3 context・最終timestepで比較するtoolを準備し、package BF16 embedding行と既存hybrid fixtureがbit-exactであることをCPU-only preflightで確認する。
3. 承認後のみ、RPB/visible-device/fused-kernel guardをprocess起動前に固定した単発R9700 windowでPhase 3cを実行する。今回見つけた候補の修正はPhase 4承認まで書かない。
