# Phase 32: native FP8 KV append encode再検証

> 状態: 完了（exact gfx1201 native scalar限定採用）
> 作成日: 2026-08-19

## 完了結果

exact `gfx1201`のnative scalar／packed prototypeはsoftware baselineとbit exactだった。2026-08-19のユーザー指示で固定5%規則を
担当AIの理由付き裁量判断へ置き換えて再評価し、既存kernel/workgroup/symbol/ABIを変えずcompile-time helperだけをnative化できる
C1 scalarをdynamic/static FP8 appendへ限定採用した。C1は全operator rowを19.40〜65.22%、production 10,001-token append familyを
51.52%短縮した。C2 packedは追加workgroup/store/tail複雑性に対してfull-model寄与が小さいためprototypeに留めた。
10,013-token OpenAI non-stream/SSEとshutdown cleanupもPASSした。default FP16、gfx1030 software route、public API、KV formatは変更していない。

機械可読なidentity、operator/full-model evidence、制限は
[Phase 32 bounded summary](../../../../../../ci/matrix/phase32-native-fp8-append-summary-v1.json)を正とする。

## 目的

Phase 30で棄却したexact `gfx1201`向けnative OCP E4M3FN KV append encodeを、Phase 31で成立した
10k+ input、chunked prefill、通常CLI/APIのdynamic/static FP8 KV経路で再検証する。

Phase 30のoperator比較ではchunk 1/32/256を測り、native candidateはchunk 256でsoftware baselineより
68.69%悪化した。一方、当時は10k+ full modelがworkspace preflightで実行不能であり、実運用の
selected chunk、全layer append、TTFTへの寄与を測れなかった。Phase 32はこの前提変化だけを再検証理由とし、
native FP8命令の存在自体を採用理由にしない。

主な成果は次の三点である。

1. current software appendが10k+/16,385-token prefillのdevice時間とfull-model時間に占める割合を測り、
   user-visible寄与と局所機構改善を分離する。
2. exact `gfx1201`でsoftware、native scalar、native packed pairを最大3候補として比較し、codec、reduction、
   memory storeの寄与を分離する。
3. 一般のadoption scope規則に従ってproduction採否を決定し、棄却時はtemporary candidateをproduction sourceから除去する。

## 先行事実と変更された前提

- Phase 30のnative FP8 readは全256 code bit exactで採用済みだが、append encodeはchunk 256の68.69%悪化により棄却した。
- Phase 31でrequest workspaceを約86.79%削減し、Qwen3.5-4B BF16 weight + dynamic FP8 KVについて、
  exact `gfx1201`の16,385 inputを16,384+1の2 chunk、exact `gfx1030`の10,001 inputを1 chunkで実行した。
- current appendは両target共通の`float_to_e4m3fn` software binary searchを使う。dynamic FP8はtoken/head rowごとに
  K/V最大値をreductionし、scaleをF32で保存してから各要素をencodeする。static FP8は同じkernel bodyで固定scaleを使う。
- gfx1201 attention readは既にnative conversionであり、Phase 32の性能差はappend write側だけへ帰属させる。
- default KV encodingはFP16のままである。Phase 32はlow-bit default昇格またはquality policyを決めない。

## Phase固有の範囲

### 対象

- primary target: Radeon AI PRO R9700、exact `gfx1201`、UUID `GPU-a8e9ddefa2d60f55`。
- control target: Radeon Pro V620、exact `gfx1030`、UUID `GPU-76a08c022586fed6`。
- encoding: dynamic FP8をprimary、static FP8をcodec/reduction attribution controlとする。
- input: model-free append token `1/31/32/33/255/256/257/511/512/513/2047/2048/2049/9999/10000/10001/16383/16384/16385`。
- full model: fixed Qwen3.5-4B BF16 GGUF/derived lock、10,001および16,385 input、greedy 1/2 output。
- existing Phase 31 automatic chunk selector、liveness arena、vAttention virtual-contiguous KV、Phase 30 native read/wave attention。

### 非対象

- FP16/NVFP4/TurboQuantの新しいencode provider。
- low-bit KVのdefault昇格、quality benchmark、recipeまたはscale granularityの変更。
- Paged Attention、FlashAttention/matrix attention、prefix sharing、continuous batching。
- Q/K/V projectionとのfusion、RoPEとのfusion、semantic graphまたはpublic APIの変更。
- gfx1030をnative FP8と表記すること、別RDNA4 SKU・gfx1200・gfx942への実機一般化。
- Phase 30のnative attention read/wave providerを再評価すること。

## Candidateとprovider境界

### B0: current software baseline

- 256-thread block、token/head当たり1 block。
- current K/V F32 maximum reduction、F32 row scale、software `float_to_e4m3fn`を維持する。
- exact `gfx1030`はPhase 32を通してB0固定controlとする。

### C1: gfx1201 native scalar encode

- reduction、thread/block/grid、scale、store単位をB0と同じにし、最終F32→E4M3FNだけを
  `__builtin_amdgcn_cvt_pk_fp8_f32`または同じOCP contractを持つcompiler builtinへ置換する。
- native instruction前後の補正が必要なら一候補内へ閉じ、codec以外の機構を同時に変えない。
- Phase 30の旧candidateと同じ機構なら、その旨を明記してlong-contextで再現性だけを判定する。

### C2: gfx1201 native packed-pair encode

- adjacent 2要素を一threadで読み、native packed conversionと16-bit storeを使用する。
- dynamic scaleのK/V maximumは固定treeで求め、rowの実数式、scale、RNE/saturation、NaN/Inf contractをB0と一致させる。
- head dimensionのodd tail、unaligned output、row境界を明示処理し、Qwenの256だけにcorrectnessを依存しない。
- workgroup変更による効果とnative conversion効果をC1との差で分離する。

C1/C2を同時にproductionへ残さない。model-free A1/A2で一候補へ絞り、full-model matrixはB0対winnerだけにする。

## Adoption scope

stable dispatch keyだけを使用する。

| scope | target | encoding | mode | candidate |
| --- | --- | --- | --- | --- |
| S1 | gfx1201 | dynamic FP8 | prefill append、採用可能bucket全体 | C1またはC2 winner |
| S2 | gfx1201 | static FP8 | prefill append、採用可能bucket全体 | C1またはC2 winner |
| complement | gfx1030、FP16/NVFP4、非合格bucket | current provider | B0 |

contextまたはchunk thresholdをrouting keyにする場合は、final run前に境界`B`を固定し、`B-1/B/B+1`、
scope内の複数代表値、scope外baselineを検証する。個別prompt、token値、測定結果、DPM modeをkeyにしない。

## 数値・出力規則

- software contractと全byte一致するcodec置換はN0とする。
- maximum reductionを逐次/LDS treeから固定balanced treeへ変える場合、非負最大値は数学的に結合的であり、
  NaNを事前に除くcurrent contract下でbit exactを要求する。scaleまたはencode byteが変わればN0にしない。
- 全finite BF16入力、signed zero、subnormal、tie、saturation、NaN、正負Infを比較する。
- dynamic/static K/V value byte、F32 scale bit、committed length、生成tokenをbaselineと照合する。
- 原因不明のbyte/token差、non-deterministic result、scale recipe変更はN3として棄却する。
- N2変更を性能だけで採用せず、必要になった時点でユーザー判断へ戻す。

## 測定contract

- current `main`、ROCm 7.14.0、exact target、release options、Qwen3.5-4B GGUF/derived lockをidentityへ固定する。
- local Qwen serviceは停止状態を維持し、一度に一GPUだけ測定する。GPUはUUIDで単独可視化する。
- pre/during/postのforeign process、VRAM/GTT、ECC、temperature、clock、power、loader rootを記録する。
- operatorは同一binary、同一入力、同一stateでB0/C1/C2をinterleaveし、HIP event device timeを使う。
- operatorはwarmup 5以上、measured 21以上とし、median、MAD、p10/p90、bytes/tokenを記録する。
- full modelはprofilerなしで3 independent process以上、各process 3 warmup + 10 measured、
  `B-A-A-B`または`A-B-B-A`でcounterbalanceする。
- bracket baseline p50 driftが2%を超えるprocessは採否へ使わず、health確認後に一度だけ再取得する。
- append family device time、prefill/TTFT、全体timingを分離し、profiler observer effect下のwallを採否へ使わない。

## Verification matrix

### H0: host/build/ISA

- exact gfx1030/gfx1201 compile/linkとwrong-target load拒否。
- gfx1201 candidate code objectに`v_cvt_pk_fp8_f32`等のnative encode命令が存在する。
- gfx1030 code objectにgfx1201専用encode命令またはcandidate symbolが混入しない。
- provider metadata、actual dispatch symbol、target、workgroup/gridが一致する。

### G1: model-free codec・append

| case | token count | encoding | 目的 |
| --- | --- | --- | --- |
| O0 | exhaustive BF16/special/tie fixture | dynamic/static FP8 | value/scale bit exact |
| O1 | 1/31/32/33 | dynamic/static FP8 | decode/small prefill control |
| O2 | 255/256/257 | dynamic/static FP8 | Phase 30再現 |
| O3 | 511/512/513、2047/2048/2049 | dynamic/static FP8 | chunk bucket境界 |
| O4 | 9999/10000/10001、16383/16384/16385 | dynamic/static FP8 | 実long chunk scaling |
| O5 | head dim 255/256/257、start 0/1023/1024/1025/10000 | dynamic/static FP8 | tail/layout/VMM境界 |

全caseでK/V payload、scale、end position、publication、fallback false、cleanup 0を確認する。

### G2: full model

| case | target | input | KV | chunk | 主指標 |
| --- | --- | ---: | --- | --- | --- |
| F0 | gfx1201 | 4096 | dynamic FP8 | one | Phase 30/shorter control、TTFT |
| F1 | gfx1201 | 10001 | dynamic FP8 | current auto one | TTFT、append share |
| F2 | gfx1201 | 16385 | dynamic FP8 | 16384+1 | multi-chunk TTFT、boundary |
| F3 | gfx1201 | 10001/16385 | static FP8 | current auto | reduction attribution |
| F4 | gfx1030 | 10001 | dynamic FP8 | current auto | complement非悪化、software identity |

F1/F2をprimary adoption patternとし、F0/F4をcontrolにする。static FP8はscale 1.0という明示実験設定のため、
dynamic FP8の採用根拠を置き換えない。

### G3: integration

- gfx1201 CLI dynamic FP8で10k+ promptから2 token生成し、baseline/winnerのtoken、usage、audit、cleanupを一致させる。
- winner採用時だけOpenAI non-stream/SSEの10k+各1 token、`[DONE]`、shutdown request/workspace 0を確認する。
- cancel/disconnect/failure publication contractはkernel/ABIを変えない場合、既存host testと一つのGPU lifecycle spotで確認する。

## 受入基準

### Correctness・mechanism

1. candidateはO0〜O5でsoftware baselineとK/V payloadおよびscale bitが一致する。
2. actual gfx1201 code objectとdispatchにnative encode命令/symbolがあり、fallbackを使わない。
3. gfx1030とscope complementはB0へrouteし、provider identity、correctness、selection overheadにstableな悪化がない。
4. KV transaction、committed length、VMM commitment、cancel/failure、cleanup、public ABI/APIを維持する。

### Performance adoption

5. 固定改善率thresholdは使わず、担当AIが絶対短縮量、全scopeの一貫性、測定confidence、correctness、target分岐、
   実装・検証・将来保守費用、hardware-native化と将来利用価値を総合して理由付きで採否を決める。
6. append shareとfull-model timingは重要なcost/benefit情報だが、単独のhard gateにしない。
7. C1は既存256-thread kernel、grid、symbol、scale recipe、store、ABIを維持したcompile-time loweringであり、
   exact gfx1201の全operator row改善とbit exact性が保守費用を上回るため採用する。
8. C2は128-thread workgroup、packed store、alignment/odd-tail検証を追加するため、C1を越える局所利益だけでは採用しない。
9. gfx1030、FP16、NVFP4は既存software/encoding経路を維持し、native命令の混入を許さない。
10. FP8をFP16より速くすることは条件にせず、同じFP8 encodingのsoftware baselineと比較する。

### Evidence・closeout

11. raw trace、model、raw KV/full logits、生成全文を追跡せず、bounded summary/schema/test、plan/history、digestだけを残す。
12. affected host/build/GPU/full-model checks、one integration review、changed findingだけのfocused re-reviewを行う。
13. 採用時はruntime、GPU/software compatibility、numerical ledger、provenance、main planを同期する。
14. 不採用時はtemporary provider、force switch、debug timingをproduction sourceから除去し、negative resultと再検討条件だけを残す。

## 作業順序

### P32-A0: acceptance・identity・Amdahl freeze

- current source/build/model/ROCm/GPU identityを固定する。
- current B0へappend-only HIP event timingを追加したbounded evidence seamを用意し、O2/O4とF0/F1/F2のappend shareを取得する。
- Phase 30の68.69%悪化をfresh O2で再確認し、Phase 31のone/multi-chunk実行をfresh baselineとして取得する。
- F1/F2のappend shareを計算し、C1/C2の局所利益と実装費用を分離して継続可否を決める。

### P32-A1: C1 native scalar prototype

- final scale後のF32→E4M3FNだけをnative scalar/pair builtinへ置換する。
- O0/O1/O2/O4でbit exact、actual ISA、device timeを確認する。
- B0より悪い、bit exactでない、またはcompile-time限定の単純な実装に収まらない場合、C1をwinnerにしない。

### P32-A2: C2 native packed prototype

- 2要素/thread、packed conversion/store、odd tailを実装し、reduction tree差を明示する。
- O0〜O5でC1/B0と比較し、一つのwinnerへ絞る。
- register/LDS/workgroup/occupancyとGB/sを記録し、chunk sizeによる逆転理由を説明する。

### P32-A3: bounded production routing

- winnerだけをcommon KV provider selectionへ接続し、exact gfx1201/encoding/bucketをstable keyにする。
- gfx1030、FP16/NVFP4、非合格bucketは既存B0を維持する。
- graph/frontend/serviceへtarget分岐を追加せず、kernel registry/runtime内部へ閉じ込める。

### P32-A4: full-model performance・correctness

- F0〜F4を一度に一GPUだけで実行する。
- baseline/winnerのappend family、TTFT/prefill、token/state、resource、fallback、cleanupを比較する。
- 3 process counterbalanced結果からS1/S2を独立採否する。

### P32-A5: integration・closeout

- 採用時だけG3 server lifecycleを実行する。
- bounded summary/schema/test、history、main plan、関連正本文書を同期する。
- plan/historyを相互linkしてarchiveし、temporary measurement seamを除去する。

## 停止・再計画条件

- codec byte/scale差がN0で説明できず、N2/N3のまま性能測定が必要になる。
- candidateがgfx1030へ混入する、wrong-target load、fallback、GTT spill、transaction/cleanup異常が発生する。
- append shareが小さい場合はcandidateの実装・保守費用をより厳しく評価するが、それだけで自動停止しない。
- C1/C2がPhase 30のchunk 256悪化を再現し、long chunkでも一貫して改善しない。
- QKV/RoPE fusion、KV format変更、FlashAttention/Paged Attentionが必要になり、本Phaseのbounded append scopeを超える。
- verification/docsがworkの30%を超える、同じcandidateが2回棄却される、見積りが1.5倍を超える場合はvariant追加を止めnegative closeoutする。

[Phase 30計画](../../../../archive/2026/08/11-20/phase30-rdna4-native-attention-kv-optimization.md)
[Phase 31計画](../../../../archive/2026/08/11-20/phase31-chunked-prefill-memory-foundation.md)
[runtime architecture](../../../../../architecture/runtime.md)
[KV memory decision](../../../../../architecture/kv-memory.md)
[数値・出力影響変更台帳](../../../../../compatibility/numerical-output-changes.md)
[メイン計画](../../../../main-plan.md)
[対応する履歴](../../../../../history/2026/08/11-20/phase32-native-fp8-kv-append-revalidation.md)
