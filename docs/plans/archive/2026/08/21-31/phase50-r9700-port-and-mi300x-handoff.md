# Phase 50: R9700実機移植とMI300X wave64引継ぎ準備

## 状態と目的

状態は`completed-limited-adoption`（限定採用で完了）。Phase 49のV620 `gfx1030`最適化を無条件に横展開せず、共通の意味契約とtarget固有providerを分離し、
R9700 exact `gfx1201`で採否を確定する。同時にMI300X exact `gfx942`へ渡すwave64設計、ABI、selector、build契約を準備する。

Phase 50でGPU性能を実証するtargetはR9700だけとした。MI300Xはhost selectorとexact `gfx942` compile/linkまでであり、
実機7行、性能採否、`project-verified`昇格はPhase 51が所有する。R9700でllama.cpp同等へ届かないこと自体は
Phase 50の完了やPhase 51開始を阻害しないが、未達行と残差を省略しない。

## 正本と入力

- 全体計画: [main plan](../../../../main-plan.md)。
- 性能系列: [Phase 37以降ロードマップ](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)。
- GPU識別: [GPU互換性](../../../../../compatibility/gpu.md)と
  [AMD GPU互換性](../../../../../compatibility/amd-gpu.md)。
- toolchain: [software互換性](../../../../../compatibility/software.md)。
- provider境界: [runtime architecture](../../../../../architecture/runtime.md)。
- model identity: [model lock](../../../../../models/model-lock.md)。
- Phase 49 closeoutはGQA P32のgfx1030限定採用、long-prefill v2の既定不採用、HIP Graph候補の撤去、
  通常5行5/5 PASSを入力とする。同一最終binaryの全7行同等はPhase 49の成果として扱わない。

## 対象と除外

### 対象

- exact `gfx1201` 1 GPU、要求batch 1、並行sequence 1、単一active requestのR9700実機性能。
- Phase 49変更のtarget分類、gfx1201 selector、operator oracle、full-model採否、資源復帰。
- 共通source変更後のV620 focused regression。
- exact `gfx942:sramecc+:xnack-`向けcompile/link、host selector非選択、wave64引継ぎ台帳。

### 対象外

- MI300X実機実行、MI300X性能PASS、別CDNA製品や複数GPUへの一般化。
- 要求batchまたは並行sequenceが2以上のthroughput、tensor/expert/pipeline parallel。
- Phase 49で棄却したHIP Graphの再導入とlong-prefill v2のgfx1201/gfx942移植。
- gfx1030用rocBLAS solution ID、閾値、env名、wave32 kernel binaryの無検証流用。
- R9700での全7行llama.cpp同等をPhase 51開始のhard gateにすること。

## 固定する実行環境

### R9700実機

- Ubuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm build/runtime 7.14.0、
  HIP `7.14.60850`、LLVM 23、exact `gfx1201`、Code Object V6、wave32を基準tupleとする。
- buildは`CMAKE_HIP_ARCHITECTURES=gfx1201`、
  `SLLM_HIP_CODEGEN_FEATURES=co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0`、
  `-mno-wavefrontsize64`を固定し、multi-archまたはgeneric成果物を性能証拠に使わない。
- canonical GPUはUUID、BDF、`gcnArchName`を相互照合して選び、HIP indexだけをidentityにしない。既存記録の
  BDF `0000:07:00.0`、UUID `GPU-a8e9ddefa2d60f55`はpreflightで再確認し、相違時は実測値を新しい正本へ固定する。
- executableと`/proc/<pid>/maps`が解決するHIP、ROCr、math libraryを同じROCm rootへ閉じる。foreign GPU process、
  target mismatch、library root混在時は測定を開始しない。
- R9700ではpower/throttle accumulatorが`N/A`でlegacy `throttle_status`に既知の揺らぎがあるため、power単独を
  hard gateにしない。取得可能なhealth、温度、VRAM/GTT、process、ECCを記録し、欠測を数値0へ変換しない。

### MI300X compile引継ぎ

- 論理target `gfx942`、`CMAKE_HIP_ARCHITECTURES=gfx942:sramecc+:xnack-`、
  `SLLM_HIP_CODEGEN_FEATURES=co_v6,wave64,xnack=off,sramecc=on,generic_processor_version=0`を固定する。
- compile/linkまたはhost selector PASSをMI300X実機PASSへ読み替えない。gfx1201候補はgfx942で非選択を維持し、
  既存のFNUZ、`contiguous-resident` KV、model-lock contractをPhase 50で変更しない。

## Phase 49変更の移植台帳

| 分類 | Phase 49の変更 | Phase 50での扱い | Phase 51への引継ぎ |
| --- | --- | --- | --- |
| target共通 | `expected_target`伝播、semantic bundle contract、M=1 native／M>1分解、prepared completion、device property cache、DerivedContiguous、7行runner | gfx1201 host/compile契約として先に固定 | 意味、alias、lifetime、BF16-RNE/FP32 accumulatorを維持 |
| wave32候補 | residual RMSNorm、GDN projection bundle、MLP gate/up/SiLU bundle | gfx1201別selectorでfamilyごとに採否 | residualは既存wave64候補、GDN/MLPはwave64 reduction再設計 |
| wave32候補 | GQA P32、attention preprocess、linear decode/short-column | gfx1201既存providerとのprofile後に個別採否 | partition、lane ownership、block、LDS/barrierをwave64向けに再設計 |
| target依存 | short/mixed matmul、rocBLAS/hipBLAS route | gfx1201でalgorithmを再照会し、gfx1030 solution IDをコピーしない | gfx942でTensile/hipBLAS solutionを別照会 |
| 制御候補 | deferred completion、short terminal last-row | queue/lifetimeとwall-clockを別candidateとして確認 | 意味契約だけを引継ぎ、target別に再測定 |
| gfx1030限定 | scaled-prefill GEMM、short decode各env、gfx1030閾値 | profile上の必要性がない限り既存gfx1201 routeを維持 | 直接移植しない |
| 移植しない | long-prefill v2、HIP Graph | Phase 49の棄却を維持 | 前提が変わるまで再提案しない |

この表は候補一覧であり、全候補の実装を完了条件にしない。fresh R9700 profileのGPU時間と全体E2Eへの上限から、
利益の大きいfamilyだけを一つずつ扱う。各候補は`adopt-gfx1201`、`keep-gfx1030-only`、`decompose/baseline`、`reject`の
いずれかへ分類し、R9700で不採用でもgfx1030採用を撤回しない。

## 測定契約

固定Qwen3.5-4B BF16、FP16 KV、MTP/vision off、greedy、単一GPU、要求batch 1、並行sequence 1を使う。
固定llama.cppはPhase 49と同じrevision/profileを使い、GGUF bytesが同一でない比較はE1 system-equivalentと明記する。

| 行 | input/output | 反復 | 追加条件 |
| --- | --- | --- | --- |
| short odd | 17/17 | 3 warmup＋10 measured | 固定非整列token列 |
| short aligned | 32/32 | 3＋10 | 固定token列 |
| prefill long | 1,024/128 | 3＋10 | 固定token列 |
| decode long | 32/256 | 3＋10 | 固定token列 |
| long prefill | 10,001/2 | 3＋10 | token ID `23066`反復 |
| extended prefill | 100,000/2 | 1 warmup＋3 measured | context 131,072、token ID `23066`反復 |
| extended decode | 32/20,000 | 1＋3 | context 131,072、EOSと追加stop無効 |

- `engine-performance-direct-v2` schemaとrunnerを正本にし、両engineのinput token列、output budget、context、dtype、KV、
  timing boundary、engine順を固定する。通常行と長時間行で規定反復数を混同しない。
- E2E、TTFT、prefill、TPOT、token/s、median/MAD、全反復、GPU family/kernel時間、peak/resident VRAM、GTT、health、
  process、fallback、cleanupを保存する。各engine内のgenerated/visible token、stop reason、反復digest一致をhardに確認し、
  cross-engine digest差はE1観測として記録する。
- OOM、timeout、crash、0測定、途中終了をPASSや行の省略へ読み替えない。理由と取得済み証拠を固定して未達とする。

## 作業単位と順序

### P50-A: identity・preflight・fresh baseline

1. source tree、build input、ROCm root、compiler、model lock、GGUF、固定llama.cpp、runner/schemaのdigestを固定する。
2. UUID/BDF/target mapping、foreign process、health、VRAM/GTT baseline、library closureを採取する。
3. current sourceの既存gfx1201 routeでsLLMと固定llama.cppの7行baselineを取得する。
4. 行ごとのGPU family、kernel/provider、host wait、transferをprofileし、候補familyのAmdahl上限を作る。

既存のR9700 10,001/2 `3.936429665`秒対固定llama.cpp `2.063845785`秒は参考値だけとし、新baselineを置換しない。

### P50-B: 共通control-planeとselector分離

1. target共通のsemantic bundle、M=1 native／M>1分解、DerivedContiguous、prepared completion、device property cacheを
   gfx1201 buildとhost testへ通す。target共通化によってproviderを自動選択しない。
2. gfx1030、gfx1201、gfx942のselectorをRust/nativeで一致させる。target別envを追加する場合は既存`GFX1030` envの意味を
   広げず、unset/`1`、`0`、unknown、force-baselineの優先順位を明示する。
3. positive selectorはexact targetと承認shapeだけに限定し、scope外M、head、dtype/KV、boundary前後、gfx942、unknown targetは
   既存providerまたは明示errorへ送る。provider error後のCPU/別backend fallbackは作らない。

### P50-C: R9700 candidate採否

fresh profileで上位のfamilyだけを次の順序を参考に分離評価する。上位でなければ理由を記録して未実装で閉じる。

1. **GQA split**: 既存gfx1201 wave provider、P16、P32を比較する。P32はpartition数でありwave sizeではない。
   query 1、GQA4、head dimension 256、FP16 KV、KV `4095/4096/4097`と長いKV、workspace/dispatch 2を確認し、
   partition/tile/gridをR9700実測で決める。
2. **decode融合**: residual RMSNorm、GDN projection、MLP gate/up/SiLUを別candidateとしてこの順に測る。
   fixed Qwen shapeのM=1だけをnative候補とし、M>1、adapter、MTP、multimodal、sidecarは分解経路を維持する。
3. **attention/linear**: attention preprocess、linear decode pair、short-columnを別candidateにする。token `1/3/16/17/31/32/127/128`、
   position component、state commit/rollback、RoPE、BF16-RNEを確認する。
4. **matmul**: M=`8/9/16/17/31/32/63/64/127/128`と実model shapeでgfx1201 algorithmを再照会する。
   gfx1030のsolution `-445/-472/-473`は候補IDとしても流用しない。
5. **execution制御**: deferred completionとterminal last-rowはprovider候補と分離し、queue reuse、abort/Drop、cleanup、
   output digest、full-row wall-clockで採否する。

operator改善だけで採用せず、対応するfull-model行がbaselineの測定幅を越えて改善し、他行へ原因不明な重大退行を入れないことを
確認する。性能が雑音内または悪化ならtarget固有selectorを開かず、共通意味契約とテストだけを保持できる。

### P50-D: R9700 integration candidate

1. 採用candidateをまとめた単一gfx1201成果物で7行を再実行し、固定llama.cppとの差と残差profileを固定する。
2. 全行でHIP-only、fallback/partial offloadなし、finite、生成反復一致、要求後cleanup、process終了後VRAM/GTT復帰を確認する。
3. 共通sourceを変更した場合はV620の通常5行をPhase 49 closeout条件で再実行する。長時間経路へ影響する変更だけ、
   V620の`100,000/2`または`32/20,000`を追加する。gfx1201固有変更だけならgfx1030非選択testで代替できる。

### P50-E: MI300X handoff

1. Phase 50最終sourceをexact gfx942 featureでcompile/linkし、新しいgfx1201 providerがgfx942で非選択になることをhost testで示す。
2. target共通semantic contract、shape、数値順序、alias/lifetime、workspace、dispatch、rollbackをwave64引継ぎ台帳へ固定する。
3. GDN/MLPのwidth-32 shuffle、GQA P32のblock128/partition、linearの4×wave32/2×128 ownership、attention preprocess、
   matmul solutionを「直接利用不可」とし、Phase 51でwave64向けlane ownership、block、LDS/register、barrierを再設計する。
4. Phase 51用のfresh MI300X preflight、7行runner、operator boundary、必要VM tupleを列挙するが、Phase 50でGPU PASSを作らない。

## 正しさ・資源・採否条件

- 数値oracleは承認shapeに加え、非整列値、selector境界の両側、tail、有限／非有限、BF16-RNE、FP32 accumulator、
  state/KV transaction、alias、abort/Dropを対象にする。
- correctness mismatch、nonfinite、wrong target/library、fallback/partial offload、crash、resource leak、foreign process、
  ECC/health異常は該当runをPASSにしない。安全なbaselineへ戻せるcandidateは不採用として他の独立candidateを継続できる。
- candidateがbaseline比でMADを越えて悪化する、wave32前提が成立しない、selectorが曖昧、または長時間行だけOOM/timeoutになる場合は、
  同じ実装を反復せずtarget分離または再計画する。
- 同じ作業単位が2回reject、review時間が実装時間超過、1時間以上機能進捗なし、検証・文書が30%超、見積り1.5倍超、
  gate変更時は新しい候補や検証を増やさず、進め方を見直す。

## 完了条件

- Phase 49変更をtarget共通、gfx1201採用、gfx1030限定、baseline/decompose、不採用、gfx942 wave64再設計へ分類済みである。
- exact gfx1201の最終7行について、規定反復、正しさ、HIP-only、fallback、資源、固定llama.cpp差、未達理由を記録している。
- 共通source変更の影響範囲でV620 Phase 49 closeoutを維持し、gfx1030 P32既定経路を原因不明に退行させていない。
- exact gfx942 compile/linkとhost selector非選択がPASSし、MI300X実機未検証を明示したwave64引継ぎ台帳がPhase 51の入力になっている。
- 採用source/build/model/peer identity、raw evidenceの外部保存先、追跡済み要約、採否、既知制約をmatching historyへ固定している。

全7行llama.cpp同等は目標と報告項目であり、Phase 50完了またはPhase 51開始の必須条件ではない。

[全体計画](../../../../main-plan.md) / [対応する履歴](../../../../../history/2026/08/21-31/phase50-r9700-port-and-mi300x-handoff.md)
