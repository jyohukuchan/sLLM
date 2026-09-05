# Phase 76以降: Qwen3.8-27B NVFP4優先ロードマップ

> 状態: Phase 76〜78完了。Phase 78は2026-09-05のユーザー承認により旧目標の未達・未実施を記録して終了。Phase 79〜81は未完了。
> 作成日: 2026-09-03

## 現在の完了判断

[Phase 78完了記録](../../../../archive/2026/09/1-10/phase78-accepted-closeout.md)と
[履歴](../../../../../history/2026/09/1-10/phase76-78-qwen38-nvfp4.md)を現在の終了判断とする。
以下の旧gate・未完了というcheckpoint記述は当時の履歴であり、最新のユーザー判断を上書きしない。

## 目的と優先順位

2026-09-03のユーザー指示により、次の最優先目標を、手持ちGPU上で
`unsloth/Qwen3.8-27B-NVFP4`を実用的な単一要求速度で文章生成できる状態とする。
この目標を満たすまでは他精度の横断最適化とGPU batchingを開始しない。目標達成後は、残る精度の
単一要求最適化を一通り閉じ、その後にNVFP4 batchingを行う。

番号上の既定順は次とする。

1. Phase 76: exact artifact統合、正しさ、baseline/profile。
2. Phase 77: mixed NVFP4 modelの単一要求decode最適化。
3. Phase 78: 同modelの単一要求prefill最適化。
4. Phase 79: static FP8 KV、MTP、長めの実入力、CLI/APIを含む実用closeout。
5. Phase 80: 他精度の残る単一要求最適化。
6. Phase 81: NVFP4のGPU batching最適化。

Phase 76〜79の途中で一般的なFP8 artifact互換、vision、tensor parallel、continuous batchingへscopeを
広げない。Qwen3.8 artifact内に実在する限定FP8 recipeは対象modelを動かすために扱うが、これを汎用FP8対応とは呼ばない。

## 固定する対象artifact

- repository: `unsloth/Qwen3.8-27B-NVFP4`
- planning時点の`main`: `57926baca9a82b4d6906b43f2750d55315f5b10f`
- source format: safetensors + `compressed-tensors` mixed-precision recipe
- main tensor file: `model.safetensors`、HTTP content length `22,568,192,096` byte
- MTP companion: `model_mtp.safetensors`、HTTP content length `849,400,392` byte
- index inventory: 1,968 tensor entry
- architecture identity: `Qwen3_5ForConditionalGeneration`／`qwen3_5_text`
- text shape: 64 layer、hidden 5,120、intermediate 17,408、vocab 248,320、
  full attention 16 layer、linear attention 48 layer

実装開始時にはbranch名ではなく上記完全SHAと全使用fileのSHA-256、size、source rangeをmodel lockへ固定する。
planning時点のSHAは実装開始時に再解決し、変化していた場合は勝手に新しいrevisionへ追随しない。

## mixed-precision inventory

公開configとindexから確認した文章modelの主要linear recipeは次のとおりである。

| consumer | 数 | weight / activation | scale |
| --- | ---: | --- | --- |
| layer 0〜55 MLP gate/up/down | 168 | NVFP4 W4A4 | block 16 E4M3FN scale、weight/input global scale |
| 16 full-attention layerのq/k/v/o | 64 | FP8 W8A8 | weight channel、activation dynamic token |
| 48 linear-attention layerのqkv/z/out | 144 | FP8 W8A8 | weight channel、activation dynamic token |
| layer 56〜63 MLP gate/up/down | 24 | FP8 W8A8 | weight channel、activation dynamic token |
| `lm_head` | 1 | FP8 W8A8 | weight channel、activation dynamic token |

FP8 weightは合計233本である。embedding、norm、linear-attentionの`in_proj_a`／`in_proj_b`とstate parameter、
vision tower、MTP 15 tensorなどはBF16または非量子化parameterとして残る。KV recipeはstatic tensor FP8である。
したがって最終性能はNVFP4だけでなく、FP8 projection、BF16 GDN／normalization、full attention、static FP8 KV、
248,320-wide FP8 `lm_head`にも依存する。

## Phase 76: model統合とbaseline

実装済みの基盤範囲は、固定revisionのconfig/index/header identity検証、main/MTP safetensorsの範囲検証、
NVFP4 168本・FP8 233本・BF16を含む1199論理tensorのinventory、直接source load plan、mixed graph、
FP8 BF16-channel-scaleのF32 resident化、NVFP4のvalue/block/global-scale uploadである。
CLIのsafetensors直接指定、static FP8 KVのscale materialization、MTP接続はPhase 79の実用closeoutへ残す。
実モデルの初期GPU smokeはPhase 76〜78でFP16 KVを使って完了している。

基盤検証は `cargo check`、`cargo test -p sllm-core --lib`（532 passed、20 ignored）、
exact `gfx1030` HIP compile-only/public-runtime build、NVFP4 selectorのhost fault test（PASS）まで完了している。
さらに `crates/sllm-hip/tests/phase76_qwen38_actual_gpu.rs` の ignored smoke で、実artifactを常駐させた
V620 `gfx1030` の full-model prefill/decode、replay、HIP-only/fallback-free、cleanup まで確認済みである。

### 2026-09-03 actual GPU evidence

固定artifact `/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4` を、FP16 KV、17-token
prefill、4-token sequential decode、同一prompt replayで実行した。両V620とR9700はsingle-GPU resident、
HIP-only、fallback 0、cleanup 0でPASSした。R9700は全GPU可視時のphysical index 2ではCode Object V6の
`invalid image`になるため、`ROCR_VISIBLE_DEVICES=2`（または`HIP_VISIBLE_DEVICES=2`）で単一GPUとして可視化し、
HIP index 0で実行した。

| target | device | resident | prefill | decode | dispatch | fallback/cleanup |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gfx1030 | 0 | 21,650,549,568 B | 3,109 ms / 17 | 6,873 ms / 4 | 6,905 (NVFP4 decode 1,344) | 0 / 0 |
| gfx1030 | 1 | 21,650,549,568 B | 22,318 ms / 17 | 6,592 ms / 4 | 6,905 | 0 / 0 |
| gfx1201 | 0 (single-visible physical 2) | 21,650,549,568 B | 2,631 ms / 17 | 536 ms / 4 | 6,905 (NVFP4 decode 1,344) | 0 / 0 |

decode audit では `matmul.nvfp4.w4a4.block16.decode.v1`（kernel id 58）の実dispatchを確認した。なお
R9700の全GPU可視physical index 2はHIP最小kernelでも`invalid image`（gfx1201 code object）となるが、
単一GPU可視化では同じkernelとQwen3.8実モデルが正常に動作する。したがってこれはモデル実装の不具合ではなく、
現行runtimeのmulti-architecture device enumeration条件として記録し、R9700のsingle-visible結果を採用する。

### 実装順

1. Qwen3.5-27BとQwen3.8-27Bのconfig、tensor namespace、GDN、attention、RoPE、tokenizer、chat templateを差分監査する。
   同じarchitecture文字列やshapeだけを理由に既存model lockへ別revisionを混ぜない。
2. exact compressed-tensors recipeをcontainer-neutralなtensor recipeへlowerする。
   groupの正規表現優先順位を解決し、末尾8層MLPをNVFP4ではなくFP8へ確実に分類する。
3. NVFP4 `weight_packed`、block scale、`weight_global_scale`、`input_global_scale`と、FP8 value／channel scale、
   BF16 tensorをversioned GGUF recipeへlosslessに変換する。scaleとinverse-scaleの意味を名前から推測しない。
4. 既存Qwen3.5-27B graph shapeを再利用しつつ、Qwen3.8 model identity、mixed binding、static FP8 KVを追加する。
5. まずFP16 KVでweight/activationを分離したoracleを通し、次にartifact指定のstatic FP8 KVを通す。
6. exact `gfx1201` R9700を最初の実GPU targetとして切り分け、動作後にexact `gfx1030` V620へ移す。
   V620を使う際は通常運用中のlocal Qwen serviceを停止して2基を解放し、GPU作業中はlocal Qwen subagentを使わない。
7. short、512、2,048、9,435 tokenのprefillと128-token decodeをprofileし、format、consumer、shape、dispatch、
   activation quantization、attention/GDN、host waitごとのwall/device時間を集計する。

### 完了条件

- exact artifactからGGUFを生成し、全使用tensorのrole、shape、dtype、scale、range、hashをfail-closedに検証する。
- NVFP4 168本、FP8 233本、BF16が意図したproviderへ入り、weight側のBF16展開や別precision fallbackが0である。
  static FP8 KVのmaterializationはPhase 79のcloseout条件とし、Phase 76〜78ではFP16 KVを明示的rollbackとして使う。
- 非整列境界を含むoperator oracleと、固定promptのlogit/token replayを通す。
- R9700とV620でsingle GPU residentとなり、GTT spillなしでbounded single-request generationとcleanupを完了する。
- target別baseline profileからPhase 77の上位bottleneckを確定する。

## Phase 77: 単一要求decode最適化

実装済みの基盤範囲は、NVFP4 W4A4のM=1専用kernel（grid=N）とselector、provider契約、symbol/grid host検証である。
FP8 W8A8は既存providerをmixed graphへ接続し、gfx1201/gfx1030の実モデルbaselineを取得した。
必要な専用GEMV選択はこのbaselineのprofileを根拠に次の候補として確定する。

### Phase 77開始時の固定baseline

V620とR9700（single-GPU visible）で最初の実モデル測定を完了した。速度目標値はhard gateにしない。次の比較は
同一artifact・同一prompt・同一single-GPU・FP16 KV・greedyで、M=1の各projectionについて provider、kernel ID、grid、
dispatch数、TPOT、resident/peak memory、fallback、cleanupを記録する。MTP・static FP8 KV・batch>1はこの比較から除外する。

M=1を独立laneとして扱い、profile上のwall寄与が大きい順に着手する。開始時点の既定候補順は次である。

1. **NVFP4 W4A4 MLP**: gate/upの同一BF16 inputを一度だけblock-16 NVFP4へ量子化して共有し、packed E2M1、
   block scale、global scaleを直接消費するGEMVでvector load、scale broadcast、multi-column reductionを評価する。
   down projectionは別activationとして独立に扱う。
2. **FP8 W8A8 projection**: full/linear attention、末尾8層MLP、`lm_head`のdynamic token quantizationを共有可能な
   sibling projection単位で再利用する。gfx1201はhipBLASLt solution／launchを、gfx1030はpersistent BF16 weightを作らない
   byte-decode／half2経路を別providerとして最適化する。
3. **BF16・非projection**: GDN state、norm、residual、full attention、samplingはprofileで支配的なものだけを扱う。
4. kernel単体の改善後に必ずtarget-only full-model TPOTへ戻し、局所改善がwallへ転化しないcandidateは既定化しない。

Phase 77ではMTPを混ぜず逐次target decodeを固定する。これによりkernel改善とdraft acceptanceを分離する。

## Phase 78: 単一要求prefill最適化

1. P78-P0Aでexact artifactをlocal RDNA上で実行できる他engine／対応forkの有無を先に確定する
   （2026-09-04完了、適格forkなし）。
2. P78-P0Bで有効な対応fork、固定llama.cpp、operator roofline／vendor providerを使い分け、現行sLLMの速度差、
   GPU時間、dispatch、実装構造を固定する。
3. P78-P1で比較結果からbottleneck順位、実装候補、中間replan値を更新し、改訂したPhase 78計画を確定する。
4. 改訂後の順序でNVFP4 W4A4、FP8 W8A8、BF16 GDN／Full Attention、host runtimeを実装・採否する。
5. M=`17/128/512/1024`とK/N境界のoperator、512／2,048／9,435-token full modelを測り、速度hard gateを閉じる。

### P78-P0A: 実行可能な比較engine／forkの確定（2026-09-04完了・適格forkなし）

upstream vLLM／SGLangは、固定Unsloth NVFP4 safetensorsをlocal V620 `gfx1030`／R9700 `gfx1201`で
実行するPhase 78の比較候補から外す。2026-09-04の確認では、公開されているSGLangのAMD `petit_nvfp4`経路は
MI250／MI300X／MI325Xが対象でlocal RDNAを含まず、見つかったQwen3.8-27B対応fork／recipeもNVIDIA
SM70／SM120／SM121向けだった。`vllm.cpp`の公開計画もこのartifactのmixed FP8 spellingを未loadとしており、
exact artifact、local RDNA、single GPU、fallbackなしを同時に満たす実行可能forkは見つからなかった。
したがってupstream vLLM／SGLangのinstall、build、起動試行をP78-P0で繰り返さず、対応を期待した待ちも作らない。

将来または実装着手前に新しい対応forkが提示された場合だけ、固定commit、固定artifact revisionまたはbyte-identical copy、
全層GPU、single GPU、NVFP4 W4A4／FP8 W8A8／BF16の実quantized dispatch、CPU offload／BF16 weight展開／fallback 0、
exact `gfx1030`または`gfx1201`をすべて確認して比較対象へ追加する。モデル変換、他engine側へのmodel対応実装、
tensor parallel、別GPUでの数値を同一GPU比較の代用にはしない。該当forkがない状態は比較作業のblockerではなく、
直ちにP78-P0Bの代替比較へ進む。

upstream vLLM／SGLangおよびNVIDIA向けforkはE2E速度比較には使わず、FP8、GDN、attention、fusion、graph／dispatchの
実装構造を調べるno-copy referenceにだけ使う。異なるGPUで公開されたthroughputは文脈用参考値であり、Phase 78の
速度差、目標値、candidate採否へ代入しない。

### P78-P0B: 実装前の有効な外部比較とroofline分解

新しい性能kernel、fusion、selectorを実装する前に、次の三層を分離して現行sLLMと比較する。

1. **system-equivalent E2E**: 固定llama.cpp build 901／commit
   `4df29be4f4c3673f428170fda944a5b19f743bb8`を同じGPUで測る。Q5_K_XLでartifact bytesが異なるため
   E1 system-equivalentであり、strict-identicalとは表記しない。
2. **exact-artifact profile**: sLLMの固定Unsloth artifactをoperator familyへ分解し、NVFP4、FP8、BF16 GDN／
   Full Attention、quantization、KV、elementwise、host waitのwall／device時間と実trafficを測る。
3. **operator ceiling**: 同じM/K/Nのbest available vendor／reference provider、計算roofline、memory rooflineを使い、
   NVFP4 decodeは実効read帯域、NVFP4 prefillはunpack／scaleを含む実測と計算上限、FP8／BF16は利用可能な
   hipBLASLt／rocBLAS等のproviderとの差を記録する。精度やtrafficが異なる値は上限または下限と明記し、
   strictなpeer性能とは呼ばない。

P78-P0Aの条件を満たすforkが後から得られた場合だけ、上記にstrict-identicalまたはrecipe-equivalent E2Eを第四層として
追加する。この層が空でもP78-P1に必要な比較は成立する。

比較はV620 `gfx1030`とR9700 `gfx1201`をそれぞれ単一可視GPU、active request 1、parallel 1、MTPなし、
greedy、同じ9,435-token prompt、同じoutput budgetで行う。公平条件はFP16 KVを第一controlとし、engineが同形式を
扱えない場合は実際のKV形式と推定trafficを明記する。既存のllama.cpp Q5_1 KV＋MTP幅3 best resultは別表の参考値にし、
公平controlへ混ぜない。内部batch／ubatch／chunkは各engineの実用既定を使えるが、値と実効shapeを記録する。

P78-P0Bは`17/17`、`512/32`、`2,048/128`、`9,435/128`を1 warmup＋3 measuredで取得する。現行sLLMの
9,435-token runが長時間となる間は、すでに取得済みのidentity一致baselineを初回比較へ再利用できるが、改訂計画の最初の
採用候補を入れる前にfresh representative runを少なくとも1回取得する。各runでE2E、TTFT、prefill、TPOT、tok/s、
peak VRAM、device memory read、GPU utilization、dispatch数、fallback、生成token digestを残す。代表runではrocprof/HIP eventを
使い、NVFP4、FP8、GDN、Full Attention、quantization、KV、elementwise、host waitのGPU/wall時間を同じfamilyへ正規化する。

性能値だけでなく、比較engineのkernel／実行構造も調べる。llama.cppは直接reuse可能なMIT sourceとしてweight streaming、
MMQ/GEMV、attention、graph／command submissionの該当commitとfileを記録する。llama.cpp以外は、実行不能な
upstream vLLM／SGLangや他GPU向けforkも含めてno-copy referenceとし、tile、activation再利用、fusion、dispatch境界、
KV accessの設計上の要点だけを独立noteへ抽出する。精度形式、MTP、KV、GPU、内部batchの差による利得と、
sLLM実装差による利得を混同しない。

### P78-P1: 比較後の計画改訂

P78-P0A/P0Bの結果から、両targetごとに累積wall時間80%以上を占めるoperator family、有効な対応engine比または
roofline／vendor provider比、期待上限、候補実装、検証shape、rollbackを表にする。実装順はwall寄与と、利用可能な
比較上限に対するgapの積が大きい順へ並べ直す。現在下記に記載する
NVFP4 matrix化、multi-column decode、activation共有、FP8、GDN／Attention、host overheadの順序は開始仮説であり、
P78-P1で確定するまで固定しない。比較でFP8、GDN、attention、host wait等がNVFP4より大きいと判明した場合は、
NVFP4を先に実装する理由に過去の予想を使わず、計画順を変更する。

改訂時には速度hard gateそのものは緩めない。変更できるのは実装順、candidate tile／fusion、operator別の中間目標、
測定頻度である。比較表、profile、改訂理由をこの計画へ追記し、P78-P1の計画差分を確定してからproduction selectorを
変更する。P78-P0A/P0B/P1が終わるまでは、新しい性能candidateを既定化しない。

#### 2026-09-04 profileによる改訂結果

P78-P0Bのexact-artifact profileを受け、開始仮説の一律なNVFP4-first順は採用しない。9,435-token prefillへ
最初のmatrix候補を入れた代表runでは、V620が`112.768 s`、R9700がrocprof込み`72.347 s`だった。GPU family内訳は
次のとおりである。比率は各代表rocprof runのkernel duration総和に対する値で、別GPU間の性能比較には使わない。

| target | Full Attention | FP8 projection | NVFP4 projection | BF16→NVFP4 quantize | 観測上の優先順 |
| --- | ---: | ---: | ---: | ---: | --- |
| V620 `gfx1030` | 64.545 s / 57.84% | 26.855 s / 24.07% | 13.993 s / 12.54% | 3.473 s / 3.11% | attention → FP8 → NVFP4/quantize |
| R9700 `gfx1201` | 51.851 s / 73.19% | 1.291 s / 1.82% | 13.065 s / 18.44% | 3.096 s / 4.37% | attention → NVFP4/quantize → residual |

既存generic full attentionはQwen3.8のQ24/KV4、すなわちGQA6を専用prefillへ送らず、同じK/Vをquery row・
query headごとに再読していた。最初のattention候補を`Q_TILE=4 × GQA6 × K_TILE=32`、FP16 KV、32 KiB LDS、
workspaceなしに固定し、operator中間値を16 full-attention層合計でV620 `<=6.92 s`、R9700 `<=3.03 s`とする。
これは完了gateではなく、未達時にK16/K32、blockwise softmax、half2 accumulationを再選別するreplan値である。

実GPU 5 warmup＋21 measuredでは、最悪の実chunk位置に相当するV620 `M512,start=8704`でgeneric
`412.159 ms`、既存GQA6 qtile4 control `78.142 ms`、K32 `76.801 ms`、R9700 `M1024,start=8192`で
`644.146/95.175/258.561 ms`だった。全経路は同じoutput digest、最大1 BF16 ULP、FP16 KV roundtrip、fallback 0、
cleanup 0を満たした。したがってK32はV620で1.7%だけ、R9700で2.72倍の退行となるため棄却する。qtile4 controlは
generic比V620 5.27倍、R9700 6.77倍なのでfull-model候補へ進めるが、中間attention予算にはまだ不足する。次は32 KiB
LDSによるoccupancy低下を避け、同じonline-softmax順序のK4/K8/K16を個別比較する。

R9700の`512/32` target-only decode代表runは`6.842 tok/s`、TPOT `146.155 ms`だった。kernel profileでは
NVFP4 ID58が`2.437 s / 5,376 calls`（全kernel時間の48.05%、約76.17 ms/generated-token）、FP8
hipBLASLtが`0.926 s / 6,400 calls`（18.26%、約28.94 ms/token）を占めた。FP8 resident weightは約
9.19 GB/tokenで、このprofileに現れた200 projection/tokenのsubset単独の実効readは約318 GB/s、R9700 nominalの
49.7%に達する。ここに含まれないfull-attention K/V 32本と`lm_head` 1本は別symbol／algorithmとして後続profileで
分離し、全233本の値traffic（約10.625 GB/token）と混同しない。したがって
R9700 decodeはNVFP4を、1 output列につき256-thread reductionするID58から、activation packed value／scaleを
workgroupで一度だけ共有して複数列をDP4A計算するproviderへ最初に置き換える。FP8 hipBLASLtは直ちにcustom GEMVへ
置換せず、NVFP4後のfresh profileで50%帯域相当を維持できるか再判定する。V620 decodeの順序は同じ測定を取得してから
独立に固定する。

この改訂後の実装順は、prefillでは(1) GQA6 K32 attention、(2) V620 FP8 half2/MMQ、(3) target別NVFP4
matrixとwave-per-block activation quantizer、(4)残るGDN／host、decodeでは(1) NVFP4 multi-column DP4A、
(2) V620 FP8 custom GEMVまたはR9700 hipBLASLt再選別、(3) attention／GDN／host残差とする。各candidateは
opt-inとrollbackを保ったままoperator oracle→512→2,048→9,435の順で判定し、数値またはwallへ転化しないものは
既定化しない。

最初のNVFP4 decode候補ID65はpacked activationとscaleを13,056-byte LDSへ置き、128 threadが128列を1列ずつ
serial-K処理した。非整列を含む21-case oracleは両targetでPASSしたが、Qwen形状の3 warmup＋10 measured中央値は
`K5120,N17408 / K17408,N5120`でV620がID58 `0.779/0.786 ms`に対しID65 `0.949/1.388 ms`、R9700が
`0.538/0.520 ms`に対し`2.789/1.035 ms`だった。したがってID65は既定化せず棄却し、次候補は1 waveが複数列を持ち、
block16 dotをlaneへsplitしてwave reductionする構造へ変更する。wave8 activation quantizerはtail書込み境界とfull-wave
shuffleを修正後、同じ21-caseを両targetでPASSしたため、性能A/Bまでopt-in候補として維持する。

R9700 ID64のISA/profile監査では、full `M1024`がID64時間の97.55%を占め、Qwen wide/downの双方が約
`21.6 TFLOP/s`だった。K32 stage当たり8 FP8 WMMAに対してblock16 scale適用のFP32 mul 128、add 64と
依存delayが残る。次のprefill候補はID64を書き換えず、E2M1×E4M3 scaleを正確にFP16 operandへ吸収し、native
FP32 accumulator fragmentをK全体で保持する別opt-inとする。Qwen operator中間値はwide/downそれぞれ
`8.460/8.366 ms`から`<=4.230/4.183 ms`、spill 0、allocated VGPR概ね128以下とし、aligned full tileでのみ
K64をK32と比較する。単純N128化や二重bufferはこのscale hot loopを消さないため先行しない。

V620 FP8 decode候補ID66は1 waveで4 output列、1 WGで32列を持ち、exact E4M3FN→FP16 half2を4列へ
再利用する。K/N境界を含む9-case oracleは最大相対誤差`0.0038395557`、fallback 0、cleanup 0でPASSした。
3 warmup＋10 measuredの中央値は`K2560,N9216 / K9216,N2560 / K5120,N17408 / K17408,N5120`で
ID6 `2.121/5.102/11.566/12.467 ms`に対しID66 `0.375/0.989/1.278/0.540 ms`、5.16〜23.08倍だった。
後半2形状は測定中のclock rampでsample分散が大きいため、full-model TPOTと安定化した再測定で採否する。ID66は
gfx1030・M1・OCP E4M3FN・K64整列だけのopt-inとし、R9700/gfx942のhipBLASLt経路を変更しない。

次のNVFP4 decode候補ID67は1 waveで4列、1 WGで32列を持ち、packed activation／block scaleをwave内で
共有してDP4A reductionする。21-case oracleは両targetでPASSし、`K5120,N17408 / K17408,N5120`は
V620でID58 `0.779/0.786 ms`から`0.252/0.313 ms`、R9700で`0.538/0.520 ms`から
`0.181/0.181 ms`へ改善した。ID67を入れた`512/32`はV620 `7.378 tok/s`、R9700
`12.292 tok/s`、fresh 9,435 profileはV620 `5.125 tok/s`、R9700 `8.381 tok/s`だった。

このfresh profileでは長文脈decodeのfull attentionがV620 `69.18 ms/token`、R9700
`46.40 ms/token`を占めたため、Q24/KV4/head256/FP16 KV専用のGQA6 P32候補を追加した。1 blockを
KV head×32 partitionとし、6 waveが同じK/V tileをLDSから共有する。9,435-token相当の5 warmup＋21 measuredでは、
既存wave splitからV620 `3.444→0.540 ms/layer`（6.38倍）、R9700
`2.938→0.474 ms/layer`（6.20倍）となった。4,096／9,435 contextの全24 query headを独立scalar
oracleへ照合し、両targetで最大1 BF16 ULP、fallback 0、cleanup 0を確認した。

V620 FP8 decode ID68はID66の4列/waveを維持し、E4M3FNを8値ずつdwordx2で読み、tableを使わない
x4→half2変換へ置き換えた。主要7 shapeのmicro加重値はID66比で概ね1.35倍、spill 0、VGPR 62である。
R9700はhipBLASLtのzero-workspace heuristicをdecodeだけ再選別し、代表`K6144,N5120`でrank 0
`134.2 us`に対しrank 7 `49.9 us`を得た。rank 7は全decode planをprepareでき、`512/32`のdecodeを
`12.292→13.230 tok/s`へ改善した一方、M>1へ適用するとprefillを退行させたためM=1限定とした。

GQA6 P32、ID68またはrank 7、ID67を合成した0 warmup＋1 measuredの探索runは、V620がprefill
`160.49 tok/s`・decode `9.994 tok/s`、R9700が`318.06/13.927 tok/s`だった。これは両GPUを同時実行した
競合ありのreplan値で最終evidenceではないが、直前の単独9,435 runに対してdecode wallをV620
`24.978→12.808 s`、R9700 `15.272→9.191 s`へ短縮した。次は単独fresh profileで残差を再順位付けし、
prefillはV620 FP8とR9700 NVFP4、decodeはactivation共有／launch削減を先行する。

単独・profilerなしのfresh 9,435/128 replan runでは、V620がprefill `56.457 s`／`167.118 tok/s`、
decode `12.950 s`／`9.884 tok/s`／`101.170 ms/token`、R9700がprefill `27.514 s`／`342.920 tok/s`、
decode `9.156 s`／`13.980 tok/s`／`71.529 ms/token`だった。両方ともHIP-only、fallback 0、cleanup 0である。
固定gateまでの残差はV620 prefill `2.039x`・decode `1.706x`、R9700 prefill `2.272x`・decode `1.507x`である。
V620単独rocprofのprefill device `59.013 s`はFP8 ID63 `26.983 s`、NVFP4 ID62 `14.127 s`、GQA6 K32
`11.800 s`、NV activation quantize `3.650 s`が主因である。decode device `10.914 s`はFP8 ID68
`5.310 s`（`41.48 ms/token`）、NVFP4 ID67 `2.812 s`（`21.97 ms/token`）、GQA6 P32 stage1+2
`1.434 s`（`11.20 ms/token`）が主因だった。R9700のwave8量子化込みreplan値も明示し、量子化selector抜けの
profile値は正式比較に使わない。次の順序は(1) R9700 NVFP4 scale-aware WMMA、(2) GQA6 prefillのblockwise
online-softmax、(3) V620 FP8 prefill構造変更、(4) decode weight stream、(5) projection bundleとHIP Graphを含む
launch削減とする。小幅なactivation pack共有単独ではV620 decodeの約1 ms/tokenしか削れないため、weight kernelと
同時に進める。

projection packのmetadata-only plannerは、固定artifactでNVFP4 MLP gate/up 56組、FP8 MLP gate/up 8組、
FP8 full-attention Q/K/V 16組、FP8 GDN qkv/z 48組、合計128組を検出した。NVFP4 pairはartifactから読んだ
`input_global_scale`のraw F32 bits一致も必須とし、down/out/b/a/lm_headや異なるactivation viewを混ぜない。
この段階ではloweringを変えず、401 quantizerを257へ減らせるruntime bundleの入力契約だけを確定した。
固定22.5 GB artifactのrelease test、focused 4 test、core unit 536 testはPASSした。

gfx1201のhipBLASLt graph-safe判定は、exact outer-vector FP8 `M1,K6144,N5120`、heuristic rank 7、
workspace 0の直接probeでGOとなった。eager warmup後のcapture／instantiateは1 graph nodeを生成し、
1,000/1,000 replay、同一device pointer上のactivation byte／F32 scale更新7点でeager BF16 bit一致、非finite 0、
distinct output 7/7、device allocation 5/5解放を確認した。したがってR9700の最初のstateless within-layer spanは
hipBLASLtを含めてよい。通常semantic submitのCompletion／timing eventはcaptureせず、request-owned execとraw launch、
一つの外側fenceでlogical audit 835 opとphysical replayを分けて数える。

### Phase 78速度hard gate（2026-09-04ユーザー決定）

Phase 78はcorrectness、実dispatch、profileを得ただけでは完了しない。固定Qwen3.8-27B mixed-NVFP4 artifactを
単一GPU・単一active requestで実行し、prefillが固定llama.cppのsystem-equivalent下限以上、target-only decodeが理論メモリ帯域の50%相当に
なった時点だけを正式完了とする。prefillは既存llama.cpp build 901／commit `4df29be4f4c3673f428170fda944a5b19f743bb8` の
Q5_K_XL＋Q5_1 KV＋MTP幅3結果をsystem-equivalent速度下限に使う。decodeは同じllama値をhard gateにせず、
27B×4.5 bit/weightの近似weight trafficに対して各GPUの理論メモリ帯域の50%を実効利用する値へ固定する。

| target | nominal帯域 | 9,435-token prefill下限 | prefill wall予算 | 128-token target-only decode下限 | decode wall予算 | TPOT上限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V620 exact `gfx1030` | 512 GB/s | 340.80 tok/s | 27.685 s | 16.86 tok/s | 7.594 s | 59.326 ms |
| R9700 exact `gfx1201` | 640 GB/s | 779.06 tok/s | 12.111 s | 21.07 tok/s | 6.075 s | 47.461 ms |

decodeの式は、1 token当たりの近似weight readを`27e9 * 4.5/8 = 15.1875 GB`として、
`target tok/s = nominal bandwidth * 0.5 / 15.1875 GB`とする。scale、mixed FP8/BF16、KV、GDN、kernel間trafficを
含まない近似なので、最終runではhardware counterから実read bytesと達成帯域も併記するが、hard gate自体は上表のtok/sを使う。
wall予算はtoken数を上記throughputで割った算術上限であり、roundingによる判定揺れを避けるため、最終PASSは
記録したtok/sで行う。Phase 78ではMTPを無効、KVをFP16、greedy、EOSと追加stopを無効、入力9,435 token／
出力128 token、model resident後のrequest時間を固定する。作業終了前に同じprompt、出力budget、single GPU、MTPなしで
llama.cppをfresh再測定し、
異なるmodel bytes／KV形式であることをE1 system-equivalentと明記する。strict-identicalに実行できる別engineが得られた場合は
その比較を追加する。llama.cppのdecode値とE2E差は報告するが、50%帯域目標を超えるdecode値をPhase 78の追加gateにしない。

固定artifactの実tensor shapeから求めた必須quantized parameter payloadは、FP8 value
`10.624696 GB`＋channel scale `0.003823 GB`、NVFP4 packed value `7.486833 GB`＋block scale
`0.935854 GB`＋tensor scale約`1.3 KB`、合計約`19.051207 GB/token`である。したがって固定した
TPOT上限だけでもparameter payloadにV620 `321.1 GB/s`、R9700 `401.4 GB/s`、各nominalの62.72%を要求する。
さらに9,435 contextのGQA6共有後FP16 KVは約`0.618 GB/token`、linear recurrent stateの最小read/writeは
約`0.302 GB/token`である。ユーザーが固定したhard gateは変更しないが、「実artifact trafficに対する50%」ではなく
`27B×4.5 bit`近似から得た数値gateであり、実装上は少なくともnominal約66%とlaunch／activation残差を同時に満たす必要がある。

短・中入力の退行を長入力平均で隠さないため、`17/17`、`512/32`、`2,048/128`、`9,435/128`の4行を両engineで
3 warmup＋10 measured実行する。各engine内で全反復の生成token、visible token、stop reasonを一致させ、E2E、TTFT、
prefill、TPOT、prefill/decode tok/s、peak VRAM、GPU family時間、median、MAD、全反復値を残す。prefill時間とTTFTは
4行すべてで`sLLM median <= llama.cpp median + max(sLLM MAD, llama.cpp MAD)`を満たすことを相対速度gateとする。
decodeは`9,435/128`で上表のtarget-only tok/sを満たすことをhard gateとし、短・中行のTPOTと全行のE2E差は
定量報告するが、別の上限を追加しない。prefillの絶対・相対gateとdecodeの50%帯域gateをV620／R9700の両方で満たし、
HIP-only、fallback 0、partial offload 0、GTT spillなし、
non-finite 0、cleanup 0であることをPhase 78の完了条件とする。model verify/load時間、MTP、static FP8 KV、batching、
tensor parallelは測定へ混ぜない。内部chunk／tileは単一prompt内の実装詳細なので最適化してよいが、測定前に固定する。

現在の9,435-token baselineはV620 8.052 tok/s、R9700 13.073 tok/sで、絶対下限までそれぞれ42.32倍、
59.59倍の改善が必要である。17-token smokeの4-token decodeはV620 0.582 tok/s、R9700 7.463 tok/sで、
50%帯域下限まで28.96倍、2.82倍の改善が必要である。この差ではscalar kernelの小幅tuningを続けず、次の実装単位で
matrix化、activation共有、fusion、同期削減まで行う。

### 2026-09-05 ユーザー判断: V620 decode最適化の終了

ユーザーの「21.66×15.445が512 GB/sの50%を超えているならV620 decode最適化は完了でよい」を反映し、
V620 decodeの追加最適化をr25で終了する。上記の旧近似payloadによる16.86 tok/s固定gateは、V620 decodeには以後適用しない。
現行r25の通常長文decode `15.444521 tok/s`と、同じbinaryのcounter計測 `21.655160 GB/transition`から、
実効read-request帯域の推定は約`334.454 GB/s / 65.32%`となる。
さらに既存artifact payloadのみの`19.051207 GB/token`で計算しても約`294.237 GB/s / 57.47%`で50%を超える。
これは仕事量と通常wall時間による実効帯域指標であり、物理DRAM utilizationの直接実測ではない。
特にcounter runは通常runと別で、計測によるcache／実行への影響を含み得るため、その区別を保持する。
今回の終了判断はV620 decode最適化だけに適用する。V620 prefill、R9700の既存条件、
正式4行3 warm＋10 measured、数値・cleanup条件、ID72のN2採用判断は変更しない。

### 実装方法の開始仮説と採否順

P78-P1で比較結果を反映して改訂することを前提に、開始候補を次のように置く。

1. **計測をoperator familyへ分解する。** P78-P0BのHIP eventと代表rocprof runで、NVFP4 projection、FP8 projection、
   activation quantization、BF16 GDN、Full Attention、norm/elementwise、KV append、host prepare/waitをwall/device時間へ
   分ける。dispatch数だけで順位を決めない。各候補はoperator oracle後に512-token full modelへ戻し、E2Eへ転化しない
   局所改善は既定化しない。
2. **NVFP4 prefillをscalar row8からmatrix providerへ置き換える。** `gfx1201`はpacked E2M1をtile内だけexact E4M3へ
   展開し、block-16の前半／後半scaleを別fragmentへ適用するrocWMMA FP8 MMAを使う。開始候補は128行×64/128列、
   K=32/64のmulti-wave tileとし、weight/input global scaleはBF16 store epilogueへ融合する。`gfx1030`はpacked E2M1を
   FP16 bit patternへ展開するvector ingress、half2 dot2、64/128行×32/64列のMMQ tileを使う。両targetともfull-weightの
   BF16 mirrorやprojectionごとの全weight展開を作らず、resident NVFP4から消費中のtileだけを展開する。
3. **NVFP4 decodeをmulti-column GEMVへ作り直す。** activation value／block scaleをworkgroup内で共有し、1列ごとに
   同じactivationを再読込する現在の経路を廃止する。4/8列×複数wave候補、packed 32/64-bit weight load、scale broadcast、
   split-K境界をtarget別に測り、M=1はprefill selectorへ流さない。
4. **FP8 W8A8をtarget別に閉じる。** `gfx1201`はhipBLASLtのouter-vector FP8を維持し、shape別algorithm cache、scale
   pointer、workspace、最終行`lm_head`を確認する。`gfx1030`の16×16 scalar FMA tileは最終経路にせず、既存MXFP8で
   実証したhalf2 dot2／multi-column構造をouter-vector scaleへ適用する。QKV、GDN、MLP、248,320-wide `lm_head`を
   別shape familyとして選ぶ。
5. **dynamic activation quantizationをsibling間で一度だけ行う。** NVFP4 MLPのgate/up、FP8 MLPのgate/up、full
   attentionのQ/K/V、GDNのFP8 qkv/zで、同じBF16 inputのpack/value/scaleをprepared activation objectとして共有する。
   exact artifactのGDN b/aはBF16なのでFP8 pack共有へ含めない。
   bundle ABIまたは同一queue上の明示的なquantize→複数matmul契約を使い、各matmul planが同じ入力を再量子化・別workspace
   化しない。down/out projectionは入力が異なるため共有しない。
6. **projection後の残差をfusionする。** prefill GDNは4 projectionの共有pack後、recurrence／gate／outをprofile順に
   fuseする。Full AttentionはQKV preprocess、RoPE、KV append、online softmaxの既存chunked providerをshape一致で再利用し、
   FP16 KVのまま9,435 tokenを速度gateへ通す。terminal `lm_head`は最後の1行だけを計算する。norm、residual、SiLUは
   memory passが上位残差になった場合だけ隣接epilogueへ融合する。
7. **host overheadを最後ではなく並行して削る。** shape/algorithm/prepared planをrequest間で再利用し、同一stream内の
   不要なwait、per-node prepare、短命workspace allocation、重複state publicationを除く。HIP Graphはdispatch列が安定し、
   profileでhost時間が支配的と確認できたbucketだけへ採用する。exact decodeの監査では1 token当たり`932` semantic
   submissionのうち、linear state 48、attention preprocess／KV append／causal attention各16、Argmax 1をeagerに残し、
   残るstateless 835 opを約65 spanへ分けられる。最初は1層内1 spanをeager warmup後にrequest-owned graph execへ一度だけ
   capture／instantiateし、1000 replay、入力値変更、eager bit比較を通す。通常submitのper-op completion/eventをgraphへ
   captureせずraw launch-only経路を設ける。gfx1201 hipBLASLtはgraph-safeがpartialなので、rank 7とouter-vector scaleを
   含む実機probeをgo/no-goにする。全span化してもdevice kernelは残るため、V620で回収可能なのはwall/device差
   `24.9 ms/token`以下であり、weight／attention kernel最適化の代替にはしない。

実装の中間replan条件も数値化する。最初のtarget別matrix providerを入れた512-token full-modelで、現baselineから10倍以上、
すなわちV620 `<=6.011 s`、R9700 `<=3.687 s`にならなければ、そのkernelのtile微調整を止めてactivation共有または別matrix
骨格へ切り替える。projectionとactivation共有を入れた2,048-token runでは最終prefill下限の70%以上、V620
`>=238.56 tok/s`（`<=8.585 s`）、R9700 `>=545.34 tok/s`（`<=3.756 s`）を要求する。これはPhase完了gateではなく、
遅い候補へ時間を使い続けないためのreplan triggerである。最終採否は常に上記4行の実モデルgateで行う。

### Phase 78実装・実測結果（2026-09-03）

NVFP4 W4A4はM=1 decode（ID58）とM>1 prefill row8/tiled256（ID59）を分離した。
prefill kernelは8行×1出力列のworkgroup、K=256 tile、LDS weight/valueとblock scale再利用を行い、
既存のpacked elementwise経路へrollbackできる。FP8 W8A8 outer-vectorはgfx1030だけ16×16/K32の
software tile（ID60）を選び、gfx1201はhipBLASLt native（ID5）を維持する。FP8のselectorは
`SLLM_FP8_OUTER_PREFILL_FORCE_BASELINE=1`、NVFP4は
`SLLM_NVFP4_W4A4_FORCE_BASELINE=1`で旧経路へ戻せる。

非整列M/K/NとM=17/128/512/1024を含むoperator oracle（NVFP4 15 case、FP8 6 case）は両targetで
HIP-only、fallback 0、cleanup 0としてPASSした（最大相対誤差 0.00389）。
固定17-token実モデルも両targetでtoken replayがPASSし、実dispatchはV620でNVFP4 prefill 336／
FP8 tiled prefill 466、R9700でNVFP4 prefill 336／FP8 hipBLASLt 2,330であった。

chunk=1,024、FP16 KV、同一residentで512／2,048／9,435-tokenを実行したprefill profileは次のとおりである。
各行は`phase78_qwen38_prefill_profile_*` ignored GPU testのwall時間であり、数値oracle、HIP-only、
cleanup 0を含む。

| target | 512 | 2,048 | 9,435 | NVFP4 prefill / FP8 prefill dispatch (9,435) |
| --- | ---: | ---: | ---: | ---: |
| gfx1030 V620 | 60,105 ms | 242,076 ms | 1,171,765 ms | 3,360 / 4,640 |
| gfx1201 R9700 (single-visible) | 36,861 ms | 146,923 ms | 721,757 ms | 3,360 / 4,642 |

このprofileは正しさ・実dispatchの証拠であって速度達成の証拠ではない。V620 340.80 tok/s prefill・33.42 tok/s
decode、R9700 779.06・41.93というllama.cpp system-equivalent参考値に対して大幅に遅いため、Phase 78は正式完了しない。
NVFP4 multi-column tile、activation pack共有、FP8/BF16 projection、attention/GDN、host dispatchを順にprofileし、
同一single-request条件で比較対象と同等以上のprefill/decode throughputへ到達することを速度hard gateとする。
static FP8 KV、MTP、CLI/APIはこのゲート通過後にPhase 79へ進める。

### Phase 78再開checkpoint（2026-09-05）

この節は別の引き継ぎ文書を作らず、実装から復元できない測定値、採否、実行構成、途中状態だけを保存する。
Phase 78は未完了であり、下記の探索値をfinal evidenceへ読み替えない。作業はユーザー指示で一時停止しており、
GPU processとsubagentは停止済み、commit／pushは未実施である。

#### 最新の長文bestと残差

固定artifact、chunk 1,024、FP16 KV、MTP／batchingなし、single requestの9,435/128を使った。
V620は0 warmup＋3 measuredの中央値、R9700は0 warmup＋1 measuredの探索値であり、final 3 warmup＋10 measuredではない。

| target | prefill | decode | TPOT | hard gate残差 | token hash |
| --- | ---: | ---: | ---: | ---: | --- |
| V620 `gfx1030` | `33,712.571 ms`／`279.866 tok/s` | `10,155.319 ms`／`12.604 tok/s` | `79.338 ms` | prefill `1.218x`、decode `1.338x` | `sha256:8fcdd815...ea0dcc` |
| R9700 `gfx1201` | `21,475.810 ms`／`439.332 tok/s` | `8,019.704 ms`／`15.961 tok/s` | `62.654 ms` | prefill `1.774x`、decode `1.320x` | `sha256:be4c5d2...63e9a1` |

両runともHIP-only、fallback 0、terminal non-finite 0、cleanup 0だった。ローカルの非永続JSONは
`/tmp/sllm-p78-v620-chain-p64-9435x128-0w3.json`と
`/tmp/sllm-p78-fixed-deferred-9435x128-0w1.json`である。

V620の上記長文bestを生成したopt-in集合は次である。

```text
SLLM_PHASE78_CHUNK_CAPACITY=1024
SLLM_NVFP4_W4A4_PREFILL_FORCE_DP4A=1
SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_ACTIVATION_SHARED=1
SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4=1
SLLM_NVFP4_ACTIVATION_QUANTIZE_WAVE8=1
SLLM_FP8_OUTER_PREFILL_FORCE_GFX1030_HALF2_64X64=1
SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_DWORD8=1
SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P64=1
SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1030_ROCBLAS_F32=1
SLLM_QWEN38_GFX1030_DEFERRED_COMPLETION=1
SLLM_QWEN38_GFX1030_KV_APPEND_ATTENTION_CHAIN=1
SLLM_QWEN38_NVFP4_PROJECTION_PACK2=1
```

R9700の上記長文bestを生成したopt-in集合は次である。

```text
SLLM_PHASE78_CHUNK_CAPACITY=1024
SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA=1
SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4=1
SLLM_NVFP4_ACTIVATION_QUANTIZE_WAVE8=1
SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P32=1
SLLM_CAUSAL_ATTENTION_GQA6_PREFILL_GFX1201_ROCBLAS_F32=1
SLLM_QWEN38_GFX1201_DEFERRED_COMPLETION=1
SLLM_QWEN38_GFX1201_KV_APPEND_ATTENTION_CHAIN=1
SLLM_QWEN38_NVFP4_PROJECTION_PACK2=1
```

V620の次回compositeでは上記へID78
`SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P128=1`、ID79
`SLLM_LINEAR_ATTENTION_GFX1030_ROW32_LDS=1`、ID82
`SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_LDS_LUT=1`を追加する。P128はcontext 8,192以上だけでP64を置換する。
このcompositeの長文runはまだないため、上表へ外挿しない。

#### ID78〜ID84と直近候補の採否

| ID／候補 | 状態 | 保存する測定・判断 |
| --- | --- | --- |
| ID78 GQA6 decode P128 | production opt-in | exact gfx1030長context。operatorでP64比`1.168x`。長文composite未測定 |
| ID79 GDN row32 LDS | production opt-in | recurrent operator `2.51x`。512/32 full-modelで約`0.84 ms/token`短縮、数値oracle bitwise PASS |
| ID80 gfx1030 NVFP4 prefill K128 | research opt-in、非採用 | standalone加重約`1.056x`だったが512-token cold full-modelは`2,132.2→2,129.9 ms`で実質差なし |
| ID81 gfx1201 WMMA 128x32 | research opt-in、非採用 | 512/32の1＋3でID64 `872.680 ms`、ID81 `895.151 ms`、`2.575%`退行。token hash一致 |
| ID82 gfx1030 FP8 decode LDS LUT | production opt-in、長文未測定 | 512/32の1＋3でID68 `2,535.304 ms`／`12.622 tok/s`からID82 `2,471.578 ms`／`12.947 tok/s`へ改善。decode wall `2.51%`減、14,914 dispatch、token hash一致 |
| ID83 gfx1201 NVFP4→FP8 staging | **実装途中・selector隔離** | standalone candidateは下記の通り有望だがproduction launch未接続 |
| ID84 gfx1030 NVFP4 decode scale LUT | probe／設計のみ | 8 exact shape、LDS FP32 LUT。wide/down加重`1.44x`、probe全体加重`2.10x`、bitwise PASS。production未実装 |
| gfx1030 FP8 prefill LDS LUT | probeのみ、次のproduction候補 | wide/down/lm_head、M=128/512/1024。全shape加重`1.299548x`、M=1024加重`1.282977x`、bitwise／BF16 oracle PASS |

ID82の非永続A/B JSONは
`/tmp/phase78_id82_v620_512x32_id68_control_1w3.json`と
`/tmp/phase78_id82_v620_512x32_candidate_1w3.json`、FP8 prefill LUTログは
`/tmp/phase78_fp8_gfx1030_prefill_lds_lut_probe_v2.log`、ID84 probeログは
`/tmp/phase78_nvfp4_gfx1030_decode_scale_lut_probe_3plus10.log`である。

ID83 standaloneはcandidate自体のstage oracle、有限性、決定性をPASSした。probe内のID64 control descriptorがheuristic 0件となり、
同一process controlを取れなかったため最終表示は`N0`である。以下の比較は既に取得済みの同じID64 standalone値との参考比較であり、
この結果だけでproduction採用しない。

| shape | ID83 stage＋FP8 GEMM | 既存ID64 | 参考speedup |
| --- | ---: | ---: | ---: |
| wide M128／512／1024 | `0.57／0.96／1.65 ms` | `1.186／4.162／8.413 ms` | `2.08／4.34／5.10x` |
| down M128／512／1024 | `0.53／0.91／1.61 ms` | `1.339／3.945／7.782 ms` | `2.53／4.34／4.83x` |

このID83実行結果には永続ログがなく、probe sourceは
`native/hip/tests/phase78_nvfp4_gfx1201_fp8_staging_probe.cpp`にある。production worktreeは中断時点で
ID83 enum、exact-shape helper、workspace layout、staging kernelまで追加済みだが、
`launch_nvfp4_w4a4()`のID83分岐は意図的に`hipErrorInvalidValue`を返す未接続状態である。
selectorからは隔離済みで、ID83 env `SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_FP8_STAGING=1`を設定しても
既存の安全な経路へfall throughする。

再探索を避けるため、ID70 gfx1030 FP8 F16 stagingはsteadyでID71より約`39%`遅く、ID72 gfx1201 NVFP4
F16 stagingは長文でID64より約`23%`遅く、ID77 gfx1201 attention F16 tailは`21.476→21.708 s`へ退行したことも保持する。
scaled-FP8 accumulatorは約`0.56x`かつaccuracy boundary／stress失敗、native-layout／batched WMMAも不採用である。
同じ仮説を再試行する場合は構造変更または新しいprofile根拠を必要とする。

#### FLOPS feasibilityと実機注意

標準的な`2 * 27B * 9,435`ではprefillは約`509.49 TFLOP`である。最新値はV620 `15.11 TFLOP/s`、
R9700 `23.72 TFLOP/s`、hard gateはそれぞれ`18.40`、`42.07 TFLOP/s`に相当する。AMD公称peak比では
V620がFP16 vector `37.3%→45.4%`、R9700がFP16 matrix `12.4%→22.0%`またはFP8 matrix
`6.2%→11.0%`で、両gateとも理論上は到達可能である。R9700は妥当、V620はstretch targetとして維持する。
ローカルV620は`rocm-smi`上のmax powerが`250 W`で、公称TBP `300 W`より低く見えるため、V620 gateを変更する前に
local FP16／DP4A sustained rooflineを取る。理論peakだけを未達免除の根拠にしない。

#### 再開順とworking-tree状態

1. 最初にID83途中差分を読み、ID82を保持したままFP8 staging launch、hipBLASLt descriptor、reusable workspace、
   host selector／plan-freeze auditを完成する。gfx1201 operator oracle後、512/32の1＋3 A/B、正なら9,435/128を実行する。
2. ID83共有matmul差分が安定してから、gfx1030 FP8 prefill LDS LUTをproduction化する。続いてID84 NVFP4 decode
   scale LUTを別の編集単位でproduction化する。同じmatmul sourceを並行編集しない。
3. 両targetをfresh buildし、accepted config＋新候補で512/32、次に9,435/128を測る。局所改善がfull-modelへ転化しない候補は既定化しない。
4. 絶対gateを通過後だけ、17/17、512/32、2,048/128、9,435/128を両GPUで3 warmup＋10 measuredし、
   fresh llama.cpp E1、hardware counter、GTT spill、HIP-only、fallback／non-finite／cleanup 0を閉じる。
5. Phase 78完了後にだけPhase 79へ進む。

checkpoint時点のbranchは`main`、HEADは`34584f84`で、Phase 76〜78の変更は未コミット、active plan自身もuntrackedである。
ID83をselectorから隔離したcheckpoint worktreeは、gfx1030／gfx1201のrelease benchmark buildと
public host CTest（対象1/1）をPASSした。`git diff --check`もPASSしている。GPU実行は再開していない。
HIP device orderingと`rocm-smi` orderingは一致しないため、既存のR9700 evidenceどおり
`ROCR_VISIBLE_DEVICES=2`でsingle-visibleにし、logical device 0がPCI `0000:07:00.0`／`gfx1201`であることをauditする。

### 2026-09-05 checkpointからの再開

- ユーザー指示「引き続きPhase78を進め、完了して」により再開した。上記checkpointは停止時点の履歴として保持する。
- Phase 78の絶対prefill、4行の相対prefill／TTFT、target-only decode、数値／資源条件を維持する。
  Phase 79以降へ範囲を置換せず、探索runを最終証拠へ読み替えない。
- ID83の数値調査により、E2M1値とblock scaleの積をFP8へ再符号化する際の追加丸め／飽和を確認した。
  例えば`6 * 448 = 2688`が`448`へ飽和し、既存ID64のblock単位FP32 scale適用とは同じ演算にならない。
  誤差上限を保証できないN3候補としてpublic selectorからの隔離を維持し、checkpointのID83接続優先順を撤回する。
  次はID72の端数chunk選択修正、既存演算順を保持するFP8 prefill／NVFP4 decode LUT候補を進める。
  共有matmul sourceの編集担当は一人に限定する。
- 既存checkpoint binaryでV620のID78／79／82 compositeを512/32、9,435/128、1 warmup＋3 measuredで測定する。
  実行設定とbinary SHA-256、raw結果はGit管理外の`.local-artifacts/phase78-resume/`へ保存する。
- このcompositeはHIP-only／fallback 0／cleanup 0でPASSした。512/32はprefill `290.195 tok/s`、decode
  `12.795 tok/s`、9,435/128はprefill `275.312 tok/s`、decode `13.251 tok/s`（`75.464 ms/token`）だった。
  512/32のtoken hashはID82 controlと一致したが、長文はcheckpoint bestから5個目の生成tokenで分岐した。
  長文hashは`e253a3543d3ed99549324d533a3d91f54d5a2fb198468f48db83b04994b16afb`で反復内は一致する。
  単一差分の切り分けと数値分類は未完了であり、速度gate達成・最終採用とはしない。
- compositeからP128だけを外したV620 control（1 warmup＋1 measured）は長文token hashが
  checkpointの`8fcdd8157b5cf37d3aa2436f93aac1d18db18d170abada0626e6510970ea0dcc`へ戻った。
  P128が生成分岐の原因と切り分けられたが、品質劣化の有無までは判定していない。P128の採用を保留する。
- R9700のP64候補（1 warmup＋3 measured）は9,435/128でprefill `522.150 tok/s`、decode `16.739 tok/s`、
  HIP-only／fallback 0／cleanup 0だった。これは探索値であり、過去runからのprefill差をdecode専用P64の効果とはしない。
- ID72は9,435入力のうち9個の1,024-token chunkに選ばれる一方、最後の219-token chunkが遅いID59へ落ちていた。
  aligned chunkはID72、非整列chunkは既存ID64とするselector修正のfocused host testはPASSした。
  GPU build時のresearch branchによるスコープ外参照を修正し、gfx1201 release buildもPASSした。
  修正後の1 warmup＋3 measuredは17/17 `72.335／18.505`、512/32 `1202.230／16.296`、
  9,435/128 `1134.906／16.688 tok/s`（prefill／decode）だった。長文prefillは`8.313 s`で絶対下限を超え、
  ID59 dispatch 0、ID72 7,560、端数ID64 336、HIP-only／fallback 0／cleanup 0を確認した。
  ID72はFP16 ingressがexactでもfull-K FP32累積順が変わるため、数値分類と最終比較は未完了である。
- 上記探索値までのbenchmark v2はprefill末尾の生成tokenを数えず、その後にbudget回decodeしていた。
  指定された合計出力budgetに合わせ、v3では最初のprefill tokenを生成列に含め、残りbudget-1回だけdecodeする。
  TPOT／decode tok/sも実際のtransition数で計算し、絶対速度下限は変更しない。v2のtoken hash／E2Eをv3へ読み替えない。
  fresh llama比較も`n_predict=budget`に固定する。比較補助は`scripts/dev/phase78_compare.py`で、数値分類／GPU資源証拠は別途必要である。
- ID72後のR9700 kernel traceはdecodeのGPU時間`55.812 ms/token`のうちFP8 matmul `22.989`、
  NVFP4 matmul `19.794`、attention `5.882`、FP8 quantize `2.057`、GDN `1.758`、その他`3.005`だった。
  profiler付きrunのwall値を速度判定に使わず、LUTとlaunch削減の優先順位付けに使う。
- 抽出ID85の独立GPU probeは別V620（PCI `0000:03:00.0`）でtiny FP32 oracle、ID71 bit一致、
  M=17/127/128/129/219/512/1024かつK70/N65のtail全件をPASSした。抽出ID84も同GPUで
  scale finite 254 code／NaN sentinel、K48/N37 oracle、実artifact wide/down control bit一致をPASSした。
  production接続と同じPCI `0000:43:00.0`での最終性能採否は未完了である。
- fresh llama.cppは両GPUで固定fixture／合計出力budget／MTPなし／Q5_K_XL＋Q5_1 KV、4行すべて3 warmup＋10 measuredを完了した。
  全反復の生成token／text／stop reasonが一致し、server logで66/66 layer GPU offload、終了code 0を確認した。
  rawは`.local-artifacts/phase78-resume/llama-{v620,r9700}-fixed128-3w10m/`にある。

  | target | row | prefill tok/s median | decode tok/s median | TTFT ms median |
  | --- | --- | ---: | ---: | ---: |
  | V620 | 17/17 | 72.150 | 18.686 | 236.376 |
  | V620 | 512/32 | 345.108 | 19.040 | 1484.623 |
  | V620 | 2048/128 | 358.799 | 19.421 | 5709.633 |
  | V620 | 9435/128 | 350.487 | 18.292 | 26923.412 |
  | R9700 | 17/17 | 89.700 | 22.963 | 190.275 |
  | R9700 | 512/32 | 810.175 | 23.666 | 632.906 |
  | R9700 | 2048/128 | 878.681 | 24.239 | 2332.347 |
  | R9700 | 9435/128 | 829.479 | 23.171 | 11378.377 |

- hardware read counterの小規模probeでは、gfx1030の`GL2C_EA_RDREQ_32B/64B/96B/128B`から非ゼロのEA read requestを取得した。
  gfx1201の`32B/64B/128B`は同じ実kernelを29 dispatch観測しても全値0だったため、測定不能として扱う。
  counter値はSDKの`FETCH_SIZE`に対応する実効EA read bytesであり、物理DRAM bus bytesと同一とは主張しない。
  whole-model最終counter証拠はまだ取得していない。
- fresh HIP identityは全可視device 0=`0000:43:00.0`／gfx1030、1=`0000:03:00.0`／gfx1030、
  2=`0000:07:00.0`／gfx1201だった。今回のV620は前者で、既存Phase X llama比較runnerも同じPCIを指定している。
  続くrunでは単一可視deviceのPCI／targetとbinary hashを起動前に照合し、sysfs VRAM／GTT／busyを100 msで記録する。
- R9700のNVFP4 decode scale LUT転用probeは、既存probeを実験用local copyでgfx1201／ID67 familyへ変更して実行した。
  非整列oracle、scale／block oracle、control bit比較をPASSし、112 wide＋56 downの合計中央値はdirect `29.470 ms`から
  LDS FP32 LUT `21.807 ms`へ改善した（`1.351x`）。他の6 shapeは比較用でありexact artifactのNVFP4 inventoryに含めない。
  rawは`.local-artifacts/phase78-resume/nvfp4_gfx1201_scale_lut.log`、production採否は未実施である。


- ID84/85のproduction接続を再開し、ID84 gfx1030の起動時dynamic LDS指定が0だった不具合を修正した。
  activation共有領域は既存ID73と同じ`K/16 * 5 * sizeof(uint32_t)`を確保し、scale LUTはstatic LDSに置く。
  LUT初期化は有限値／NaN表現を照合済みの定数表を使う。ID84 host selector検査の環境変数と期待値の不整合も修正し、
  `sllm_public_runtime_host_test`はPASSした。productionの警告をerrorにする設定で型変換を明示した後、
  gfx1030／gfx1201のHIP release buildは両方PASSした。その後、共有量子化bundleのcompute-grid whitelistへID84を追加し、
  host launcherの`__gfx*__`（device pass専用）分岐をbuild targetによる分岐へ修正した。
  ID84有効時の公開projection-pack APIは両GPUで独立数値oracleと一致し、反復決定性・cleanup 0を確認した。
  ID85の未整列weightをpacked loadする問題も修正し、offset 1/2/3の独立FP32 oracleをgfx1030でPASSした。
- v3-r2のLUT候補は17/17、512/32、9435/128、1 warmup＋3 measuredを両GPUで完了し、controlと生成tokenが一致した。
  長文prefill／decodeはV620 `261.344／13.004 tok/s`、R9700 `1133.053／16.686 tok/s`だった。
  同一build 512/32 A/BではV620 prefill `1777.504→1845.994 ms`、decode `2417.556→2421.385 ms`、
  R9700 prefill `424.654→424.526 ms`、decode `1913.198→1912.073 ms`で、LUTだけの性能改善を確認できなかった。
  この版は性能改善として採用せず、古いarchive／clone controlのmicro結果を現行productionの性能根拠へ使わない。
  次は現行公開APIのcontrolと候補を直接比較し、NVFP4の有限scaleで丸めを変えないFMAとgfx1201 signed/signed dot4を評価する。
- R9700のhardware counterは、計測中だけ`profile_standard`へ固定するとbase read requestが非ゼロへ戻った。
  stockの32/64/128-byte binは0のままだったが、AMD upstream PR8022のgfx1201 event 149（256-byte request）を
  extra counterとして追加すると29 dispatchで`7,701,100`要求を取得した（`1,971,481,600` GL2C/EA read bytes）。
  rawは`.local-artifacts/phase78-resume/id84-counter-stable-256-gfx1201/`、設定は
  [Phase78 counter定義](../../../../../../scripts/dev/phase78-gfx1201-counters.yaml)。各測定後に`auto`へ復帰した。
  これは小規模probeの取得経路検証であり、物理DRAM bus bytesやwhole-model最終証拠ではない。
  最終counter runは256-byte binも含め、通常の性能測定とは設定・結果を分けて記録する。
- v3-r3ではID84の有限scaleで丸めを変えない明示FMAとgfx1201 signed/signed dot4を接続し、
  ID85は無効へ戻した。512/32と9435/128を各1 warmup＋1 measuredで両GPUとも正常終了し、
  反復token一致、non-finite 0、HIP-only、cleanup 0を確認した。長文prefill／decodeは
  V620 `277.454／13.109 tok/s`、R9700 `1160.956／16.604 tok/s`で、decode速度下限は依然未達である。
  R9700のprefillは採用判断待ちのID72を含む探索値であり、最終採用証拠ではない。
  rawは`.local-artifacts/phase78-resume/{v620,r9700}-v3-r3-id84.json`。
  単体probeの高速化を実モデル改善へ読み替えず、次の単位は演算順を保持するV620のFP8／NVFP4一時変換と
  P64 attentionのFP16 LDS格納を、変換費用を含む現行controlとの比較で評価する。
- P64 attentionのgfx1030 K/V LDSをFP32からraw FP16へ変更し、32 KiBから16 KiBへ削減した。
  分割・加算・online softmaxの順序は維持し、旧productionとの8条件比較と新archiveへの再リンクの両方で
  BF16 bitwise一致、FP64 oracle finite、反復一致をPASSした。v3-r4の9435/128（1 warmup＋3 measured）は
  prefill `276.444 tok/s`、decode `13.378 tok/s`（TPOT `74.747 ms`、MAD `0.130 ms`）で正常終了し、
  v3-r3と生成token一致、cleanup 0を確認した。絶対速度下限は依然未達である。
- V620 NVFP4 prefillの一時int8展開は、数値一致したがstage込み総時間が全caseで退行したため非採用。
  FP8 prefillの一時FP16展開は、consumerのpacked word viewを修正した版が現行ID71へbitwise一致し、
  K6144/N5120のM128/1024でstage込み `0.811／0.807`倍の時間となった。tinyは退行するため、
  ID86のshape限定opt-inとして接続中であり、実モデルの速度改善はまだ未確認である。
- ユーザーの「goal・作業を一時停止して」により、ここで作業を停止した。サブエージェントを停止し、
  実行中のPhase78 commandがないことを確認した。直近のgfx1030/gfx1201 release buildは両方完了済み。
  現在のsourceにはID86 FP8 FP16 tile staging、gfx1201 ID84 P4先読み、RDNA attentionのnative FP16変換を含む。
  native FP16変換は両GPUの全65,536符号でbitwise PASS。新archiveへのID86／ID84 P4の再リンクG1と、
  これらを合わせた実モデル測定は未確認なので、再開時はその結果・実行状態から確認する。
  V620 ID84 P2先読みはprobeのみで未統合（単体約1.095倍）。probe報告のPCIとROCR indexに不整合があり、
  exact PCI記録の訂正が残る。ID62 prefill FMAは数値一致したが全caseで退行したため非採用。
  最新の完了した実モデル証拠は前述v3-r4であり、Phase78は未完了、commit/pushなし。
- その後のユーザー指示「再開して」により作業を再開した。停止時のrelease buildと実行プロセス不在を確認し、
  最新gfx1201 archiveへのID84 P4再リンクG1はtiny oracle／wide／down bitwiseをPASSした。
  ID86の最新archive接続G1と、固定binaryを用いたv3-r5の実モデル測定を進める。

- v3-r5は両GPUで17/17、512/32、9435/128の各1 warm＋3 measuredを完了した。
  長文のprefill／decodeはV620 `265.213／13.700 tok/s`、R9700 `1131.527／17.607 tok/s`。
  同一入力の生成tokenは既存controlと一致し、runtime cleanup成功を確認した。これは探索測定であり、
  全4行の3 warm＋10 measuredによる最終証拠ではない。R9700のID72はN2判断待ちのopt-inを含む。
  V620はID86を含めるとprefillがv3-r4の`276.444 tok/s`から退行したため性能採用を保留し、
  同じ固定binaryの512/32でID86 on/offの実kernel profileを比較する。
  17-tokenのTTFTもV620 `364.262 ms`、R9700 `261.618 ms`でllama.cpp比較条件に未達。
- 次の作業単位を短文matmulの無効行削減とdecodeの転送削減へ絞る。
  gfx1030 FP8はM32/N32 tileがM17/K6144/N5120でcontrolの`1.421x`、
  NVFP4はM32/N64 tileがM17のwide/downで時間比`0.746／0.854`となった。
  いずれもproduction archive controlとの全BF16出力一致とtiny oracleを確認したが、
  M33では退行するため適用上限をM32とする。実モデルでの効果は未確認である。
  gfx1201 P64 attentionのFP16 LDS prototypeも8条件でFP64 oracle・BF16一致を確認し、
  合計kernel時間比`0.820`を得た。partitionと演算順を保つproduction接続を進める。
  V620 P2先読みのproduction buildは未使用変数のstrict warningで失敗したため修正中。
  P2の旧probeはROCR index 1からPCI03と訂正し、PCI43の測定とは扱わない。

- v3-r6は17/17、512/32、2048/128、9435/128の全4行で各1 warm＋3 measuredを完了した。
  両GPUとも反復決定性、既存r5の同一行との生成token一致、HIP-only、terminal nonfinite 0、runtime cleanup 0。
  V620のprefill／decodeは順に`55.932／14.746`、`290.114／12.888`、`291.950／12.557`、
  `275.936／13.647 tok/s`。短文TTFTは`334.747 ms`へ改善したが比較条件には未達。
  R9700は`72.453／19.722`、`1205.444／17.459`、`1370.493／16.920`、
  `1135.331／18.092 tok/s`。短文TTFTは`263.642 ms`で未達。
  V620のNVFP4 P2先読みは単体改善を実モデルdecodeの改善としては確認できなかった。
  R9700 P64 FP16 LDS化は長文decodeを`17.607→18.092 tok/s`へ改善した。
  sampled VRAM peakはV620 `30,161,383,424`、R9700 `30,619,774,976` bytes。
  終了後のsettled device usageもそれぞれ`21,401,600`、`59,912,192` bytesへ戻った。
- NVFP4 prefillのN128再利用tileは全4実shapeで`1.053〜1.241`倍の時間へ退行、
  M128/N32再利用tileも`1.247〜1.456`倍へ退行した。いずれも全BF16一致したが非採用とし、
  output tile探索をここで止める。R9700 FP8の16 MiB workspace候補も対象6 shapeで既存workspace 0より遅く非採用。
- gfx1030 ID82 FP8 decodeのP2先読みは、cold clockの影響を除く共通prewarmを追加すると
  233 tensor加重時間が`25.603→22.600 ms`、約`1.133x`となった。4 cold copies、全8 shape、
  非整列N37/N67、全256 E4M3 codeのoracleとproduction controlとのbitwise一致を確認した。
  同じdot／reduction順のままproductionへ組み込み、v3-r7で実モデル効果を確認する。

- v3-r7（ID82 P2統合）はV620の全4行・各1 warm＋3 measuredをPASSした。
  prefill／decodeは`56.227／15.831`、`289.591／13.808`、`292.073／13.215`、
  `275.307／14.658 tok/s`。全行の生成tokenはr6と一致し、HIP-only、terminal nonfinite 0、cleanup 0。
  長文TPOTは`73.279→68.220 ms`へ改善したが、`59.326 ms`下限には届かない。
  NVFP4 prefill scale LUTも全実shapeで`1.020〜1.060`倍の時間へ退行したため非採用。
  タイル拡大・LUT候補の追加を止め、ISAとcache内のweight再利用から作業を再計画する。
- chunk容量2048のPCI03探索は、最初の`layer.3.causal_attention`のrocBLAS workspace prepareで
  fail-closed OOMとなった。sampled VRAMは`33,244,561,408 B`まで増え、cleanup 0、終了後VRAM約17 MBへ戻った。
  runtime allocation ledgerの23.38 GBだけから約10 GBの余裕があるとは判断できない。
  既存1024の実device peakは30.16 GBで、library等のledger外使用量も含める必要がある。
  中間1536はbenchmark独自のpower-of-two validatorに拒否されたためGPU証拠ではない。
  runtimeは任意row countを扱うので、benchmarkの範囲512..8192を維持してpower-of-two制約だけを外し、
  中間1536を一度確認した。1536はOOMせずprefill/decodeを実行したが、warmupとmeasuredで
  生成tokenが変わる決定性FAILとなった。sampled VRAMは`32,778,412,032 B`、cleanup 0。
  この容量を採用せず、benchmark validatorの変更も戻して現行1024を維持する。
  2048 OOMも1536 FAILもPASS扱いせず、追加容量sweepは行わない。

- ID62のISA確認ではscratch 0、86 VGPR、6 KiB LDSを維持したまま、activation scale stagingへ
  厳密な`0.25F`係数を移すことでFP32 multiplyを115命令から84命令へ削減できた。
  1,179,904組のhost同値確認と、PCI43でのproduction対旧演算のtiny全点oracle・実shape全BF16一致をPASS。
  private referenceのscale loadとE4M3 decodeもproductionに揃え、旧archiveとの時間差が約0.5%以内であることを確認した。
  変更後はM128/M1024のwide/downで約3〜4%短縮した。
  v3-r8は全4行・各1 warm＋3 measuredをPASSし、生成tokenはr7と全行一致、HIP-only、nonfinite 0、cleanup 0。
  prefill／decodeは順に`56.284／15.934`、`294.596／13.807`、`297.307／13.174`、
  `279.797／14.684 tok/s`。長文prefillはr7から約1.6%改善したが目標未達、decodeはほぼ横ばい。
  TTFTは`328.600／1765.160／6916.961／33750.730 ms`で、全行の相対条件に未達。
  gfx1201のstrict compile-onlyもPASSしたが、gfx1201 GPU実行の証拠ではない。
- Group-M4のcache順変更は、scale loadのSLC修飾をproduction同様に外しScalarCodecへ揃えた再確認でも、
  PCI03のM128/M1024 wide/downで時間比`1.001／1.011／1.004／1.017`となり非採用。
  M65/M257のtiny全点oracleと実shapeの全BF16一致はPASSした。先の約5〜7%退行値は
  cache順だけを分離した証拠として扱わないが、比較条件修正後も性能優位はない。
- R9700 FP8のFP64 exact-sum候補は8実shapeとtiny K48/N37で数値oracle・BF16一致・決定性をPASSしたが、
  233 occurrence加重時間がrank7の`26.270 ms`に対し`103.507 ms`で全実shapeが退行したため非採用。
  candidateの実dynamic LDSは0 bytesであり、属性値の最大許容量64 KiBとは区別する。
- 同じFP8候補の有限E4M3値を`value*2^9`の整数へ厳密変換し、int64積和後にFP64 scale適用する案も、
  加重時間`106.262 ms`でrank7の`26.294 ms`より約4倍遅く非採用。追加の同系統sweepは行わない。
- v3-r8 V620の17/17 kernel profileはmeasured prefillのdevice span `310.824 ms`、
  kernel sum／union `296.639 ms`、gap `14.185 ms`、timestamp overlap 0。
  FP8の既存64x64が168 calls／`98.552 ms`、NVFP4短文32x64が168 calls／`108.061 ms`を占めた。
  FP8短文用32行tileを未適用の実shapeへ照合すると、6 shapeで約`1.2〜1.75x`、全BF16一致とtiny oracleをPASSした。
  M33は退行するためM2..32を維持し、N32/N64の適用を実測済みshapeへ拡張する。
  統合後のproduction launcher再リンクG1も11 shapeで全BF16一致・oracle・cleanupをPASSした。
  v3-r9の17/17・1 warm＋3 measuredは生成tokenがr8と一致し、HIP-only、nonfinite 0、cleanup 0。
  prefill `61.497 tok/s`、TTFT `304.616 ms`（r8 `328.600 ms`）、decode `15.924 tok/s`。
  この変更は短文だけに限定し、長文のr8測定をr9の新規実行値とは扱わない。
  R9700の同短文traceは同一queue内でもtimestamp overlapがあり、event sumとunionを混同できない。
  このunion gapからhost待ち時間を推定しない。profile付きwall値も速度gateには使わない。
- ID62で12 waves/EUを要求する候補は80 VGPR・20B scratch・6 active blocksとなったが、
  down shapeが約2%退行し非採用。HIPの`__launch_bounds__`第2引数はこの環境では
  `amdgpu_waves_per_eu`へ展開されるため、最初の値6はactive block数6を要求する指定ではなかった。
  uint32で安全に表現できるextentへ限定したindex計算の候補は、86 VGPR・scratch 0を維持し、
  wide/downで約2〜3%短縮、tiny oracle・全BF16一致をPASSした。範囲外には既存uint64計算を維持する接続を進める。
- scaled-F16 WMMAのK128/K256 group累積は、実Kに対する標準worst-case誤差bound係数が旧ID64より小さいN1案。
  最初のprivate実装はVGPR 249／scratch最大1064Bとなり約4.8〜7.2倍遅く非採用。
  追加GPU測定を止め、loop展開によるfragment生存期間の増大をcompile-onlyで調べて再計画する。
  stage loopの全展開を止めると両者116 VGPR・spill 0へ改善したため、その変更版を一度だけ測定した。
  数値oracle・決定性・nonfinite 0はPASSしたが、ID64より約2.1倍遅く、group案は非採用で確定する。
- 短文GDNのrecurrent stateをthreadごとのregister arrayへ保持するN0候補は、両GPUで
  M2/17/32/33 × zero/nonzeroの全FP32 state・BF16 output bit一致をPASSした。
  128 common prewarm＋3 warm＋10 measuredでも単体改善を確認し、exact Qwen head配置のM2..32へ接続する。
  gfx1030の212B、gfx1201の400B local scratchはコストとして記録するが、最大occupancyだけで性能不採用とはしない。

- v3-r10はIndex32 addressingと短文GDN register-stateをproductionへ接続した。
  V620全4行、R9700の17/17を各1 warm＋3 measuredでPASSし、生成tokenはそれぞれr8／r6と一致、
  HIP-only、nonfinite 0、cleanup成功。V620のprefill／decodeは17から順に
  `66.648／15.841`、`297.944／13.827`、`300.540／13.324`、`283.231／14.700 tok/s`、
  TTFTは`285.236／1747.982／6842.252／33340.634 ms`。
  R9700の17/17は`77.170／19.602 tok/s`、TTFT `247.610 ms`（setup `27.316 ms`）。
  V620 binary SHA-256は`73a283d0fe7de33ae0951439e70c1703349c8416e7c91329c3ee691e20a64413`、
  R9700は`b596a656209389d125df997e7a27d13ed1938cb513ee9216cd08f594dced88fa`。
  最終3 warm＋10 measuredの合格証拠には扱わず、Phase 78は性能条件未達を維持する。
- 次の限定変更はIndex32 NVFP4 packed loadのcache hint除去と、fresh Qwen GDN stateの
  4 planeを一つの全zero backingへまとめる初期化である。前者はPCI03の固定r10比較で
  4実shapeの全BF16 bit一致と約3.9〜7.5%の単体改善、後者はhost fault testをPASS。
  production GPU検証と実モデル性能はまだ未確認。

- r11のcompact GDN backingはR9700の48層・共通context・3 warm＋10 measuredで
  create `3.71873→1.64605 ms`、release `5.00486→2.08240 ms`。
  全4 planeの初期zero、非zero payloadのfork／source解放後export／再importを実GPUでPASSした。
  hostではbacking free失敗のorphan／poison保持と再free防止もPASS。既存cleanupがfault injectionを
  迂回していたテスト経路をhelper経由へ揃えた（productionの実hipFree error処理の欠落ではない）。
  このhelper接続だけはr11固定archive後の変更で、production fault consumeはconstexpr falseである。
- r11 NVFP4 ordinary-load productionは、別binaryで固定r10/r11を起動した7 shapeの全BF16 dumpが一致し、
  tiny M32/33/65の独立FP32 oracleもPASSした。FP8 ID71 longもpacked/scalar A/Wの4 load hintだけを変更した。
  r11実モデル測定はV620全4行とR9700短文を各1 warm＋3 measuredでPASSした。
  生成tokenはr10と一致、全実行HIP-only／fallbackなし／nonfinite 0、cleanup成功。
  V620のprefill／decodeは17から順に`66.751／15.833`、`315.276／13.764`、
  `315.517／13.269`、`296.387／14.663 tok/s`、TTFTは
  `281.070／1648.951／6515.778／31861.774 ms`。R9700短文は
  `77.378／19.629 tok/s`、TTFT `244.851 ms`（setup `25.165 ms`）。
  V620 binary SHAは`24a5666fe120bde106a596d20632e58221a8aadbf187a9409e06ea2394c965bd`、
  R9700は`4e6418eb6a383caaa9a78a29bfcc02ad017472c3a4018dfa27dc3227ec2843e7`。
  FP8 ID71もr11 productionとprivateの全7 shape oracle／BF16 bit一致をPASSした。
  Phase 78の最終性能条件は未達を維持する。

- r12はNVFP4短文ordinary-loadとFP8 decode exact M1/K5120/N10240 ordinary-loadを接続し、
  V620の17/17・9435/128を各1 warm＋3 measuredでPASSした。r11とtoken一致、HIP-only、nonfinite 0、cleanup成功。
  短文TTFT `282.405 ms`（r11 `281.070`）、長文prefill `297.333 tok/s`、decode `14.564 tok/s`
  （r11 `14.663`）となり実モデル改善を確認できなかったため、この2件だけproduction採用を取り消した。
  r11のlong ordinary-load／Index32／短文GDN／compact backingは維持する。
  r12 binary SHAは`74cb01bbb6840c8a4a3bf22421bf4dbc37488711a97377cd86edd5f9b8883a69`、
  archiveは`dec659fa9c57042f1ee3a543cdd9e038ac5b3f7e61ca5df5377c603b45126fc9`。
  両候補のproduction G1はPASSしたが、単体改善を実モデル速度達成と扱わない。
- NVFP4 decodeのraw weight load hint除去は、固定r11比較でwide約1.5%改善・down約1.8%退行、
  weighted `8.778784→8.749776 ms`（約0.33%）のため非採用、追加探索しない。
- R9700短文NVFP4のstage単位split-K=4候補は、scaled block16 termを保持したFP32部分和と固定順reductionにより、
  標準summation boundのdepthがK5120で319→82、K17408で1087→274、tiny K48で2→2となるN1候補。
  M17 wideは`947.731→968.131 us`で非採用、downは`1063.772→904.770 us`で接続を検討する。
  FP64 oracle／決定性／NaN伝播／cleanupはPASS、BF16差はwide39／down30／tiny0でreport-only。
  V620短文の同じ4分割も、固定r12に対しM17 wide `418.893→391.594 us`、
  down `684.068→531.451 us`、全shape bit差0、FP64 tiny oracle／決定性／cleanupをPASS。
  V620はexact M17の2実shape、R9700はexact M17 downだけを既存workspaceへ接続する。
  tiny形状は採用範囲外で、加算順による将来のbit差をN1条件とは別のhard gateにしない。


- r13は短文NVFP4 split-K=4、BF16 GDN thin projectionのrow-wise実行、R9700のexact
  FP8 M17/K6144/N5120 hipBLASLt rank 3（4候補取得）を接続した。両targetのproduction G1は
  FP64 oracle、反復再現性、非finite検査、cleanupをPASSした。split4のV620 wide/downは
  `401.064→392.204 us`／`758.727→499.205 us`、R9700 downは`1024.648→905.488 us`。
  BF16 thinのmeanはV620 `0.190954→0.016484 ms`、R9700 `0.178870→0.017640 ms`。
  加算順変更は標準誤差bound非増加のN1として扱い、BF16 bit一致を追加の採用条件にしない。
- r13実モデル17/17は各1 warm＋3 measuredで両target PASS。V620はprefill `79.750 tok/s`、
  TTFT `240.040 ms`（r11 `281.070 ms`から約14.6%短縮）、decode `15.827 tok/s`。
  R9700はprefill `79.014 tok/s`、TTFT `241.015 ms`（r11 `244.851 ms`から約1.6%短縮）、
  decode `19.538 tok/s`。全反復HIP-only、fallback 0、terminal nonfinite 0、cleanup成功。
  各build内の生成は決定的だが、r11との最初の差は両targetとも15番目の出力tokenで、
  V620は15→20、R9700は20→15。複数N1変更を合わせた結果であり、個別kernelへの帰属は未確認。
  binary SHAはV620 `14475a6e42f9d6edcfa106613fe6f5853cf9d1bbd22c957c93585c62d4d0472d`、
  R9700 `99351317ed7592d180cf7d405a42ac21e40df210b9eaf43c793251350bc0f3b6`。
  詳細はlocal `r13-short-split-thin-build-identity.json`と両targetの`v3-r13-short-split-thin`結果に保存した。
  この探索測定を3 warm＋10 measuredの最終証拠へ昇格しない。R9700の小さい実モデル改善と
  V620長文の残差を現在のbuildのprofileで分解する。
- R9700 FP8 native DOT4の別案は、全256 E4M3 codeとFP64 oracle／反復／cleanupをPASSしたが、
  4 cold copies、実運用shape別rank、233 occurrence加重のcontrol `17.053393 ms`に対し
  candidate `23.959728 ms`（約1.405倍遅い）で非採用。productionには接続しない。
  raw artifactはlocal `r9700-fp8-native-dot4-*`に保存し、bit一致をN0の根拠にしない。


- r13 R9700短文profileではNVFP4 split4のpartial/reduceを各56回確認した一方、
  BF16 thinは0回で実モデル経路では未選択だった。単体G1の速度をR9700モデルでの改善へ帰属させない。
  prefillの標準NVFP4 WMMAは112回／event sum `123.537 ms`で最大残差。
  全prefillはspan `254.756 ms`、event sum `236.921 ms`、timestamp union `137.705 ms`で重複があり、
  event sumやunion gapをserial device時間／host waitへ読み替えない。FP8 rank3のactual symbolは`MT128x128x32_MIWT4_4_WG32_4_1`（64回／7.544 ms）で、
  `MT16x128x32`（136回／30.724 ms）はrank0/controlだった。BF16の96 projectionは
  既存hipBLAS BF16／COMPUTE32Fの`MT16x16x32`（2.869 ms）であり、別のGDN recurrenceと混同しない。
- 次のbounded候補は、R9700短文NVFP4の32行×64列tileを8 wave（2行tile×4列tile）へ割り当てる。
  以前の非採用32行版は2 waveで4列fragmentを順次処理していたため、今回の列方向並列化とは区別する。
  K16 WMMA、scaled block term、加算順、BF16 RNEを維持し、private probeの1 geometryだけを
  固定r13 current provider・4 cold copies・common prewarm 128・3 warm＋10 measuredで比較する。
  起点はr13 profileのwide残差、提案者は担当AI、範囲はexact M17 wide/downとtiny検証、
  コストは両実shapeの単体probe、期限はこの1候補の結果取得まで。採用や新しいgateは事前に確定しない。


- 8-wave候補はwide `949.200→1038.680 us`で退行した。downのunsplit版は現行split4に対し
  `914.163→837.523 us`だったが、加算深さを274→1087へ戻すため自動採用しない。
  既存split4と同じdepth274を保つcompositionを1件だけ測り、全BF16 bit一致／oracle／repeat／cleanupをPASSしたものの
  `910.644→925.164 us`（約1.6%遅い）で非採用とした。追加のgeometry探索は行わない。
- r13 V620長文profileはprefill embedding10回（1024×9＋219）とdecode127回を識別した。
  decode event sumは`61.889 ms/token`、kernel区間spanは`69.947 ms/token`、event overlap 0。
  argmax→次embedding境界を含むspanとprofile wallの差は全体で`3.006 ms`、非kernel gap候補は
  `11.679 ms/token`（profile decodeの15.9%）だった。これはgraphが回収しうる時間の上限候補であり、
  CPU launch overheadの確定値ではない。device上位はFP8 LUT、NVFP4 LUT、causal attention、FP8量子化。
- 既存計画のrequest-owned HIP Graph spanをgfx1030のexact Qwen3.8 M1 decodeへ接続中。
  初回eager decodeで確定したprepared planとbufferを保持し、同一layer内の連続したstateless区間だけを
  capture／instantiateして再利用する。stateful演算、attention preprocess／KV append／causal attention、
  Argmax、host upload/readback、terminal projectionはeagerに残す。
  capture時に出力を実行・更新せず、replayはeventless aggregate completionを既存外側fenceで完了させる。
  logical submission／kernel identity監査を維持し、physical graph replay数・capture span数・kernel node数を別記する。
  新しいopt-inは`SLLM_QWEN38_GFX1030_GRAPH_SPANS=1`、省略がrollback。
  r14でnative/core/bridge接続と両targetのrelease buildを完了した。coreのfocused testは4/4、
  native host testは1/1 PASS。gfx1030の非整列M1K35N37 G1ではcapture時の出力不変、
  5入力のeager／graph／数値oracle bit一致、1,000 replay、早期解放BUSY、cleanup 0を確認した。
  G1 archive SHA-256は`454d1bba2796fdd589c4f8476b06ffc5726c69596fd1cc2058cc7c8c04166aad`。
  実モデル17/17の同一binary off/on（各1 warm＋3 measured）は生成token、logical submission、
  kernel identity/countがすべて一致し、HIP-only／nonfinite 0／cleanup 0をPASSした。
  Graph有効時は128 span／1,176 kernel node、初回eager後の15 decodeで1,920 replay。
  作成費用を含むTPOTは`62.693→63.487 ms`（約1.3%遅い）、TTFTは`236.207→235.055 ms`で、
  短文の速度改善は確認できなかった。長文9435/128も全token／logical kernel監査が一致し、
  TPOT `68.094→67.186 ms`、decode `14.686→14.884 tok/s`（約1.35%改善）、HIP-only／cleanup 0を確認した。
  長文向けopt-in候補だがPhase 78の最終性能条件は未達で、Graphだけでは不足する。
  提案起点は上記profile、担当AIによる実装候補、コストはnative/core/bridge接続とfocused G1／実モデル比較、
  期限はこの固定候補のA/Bまでとし、Graph実装自体をPhase 78の追加完了条件にはしない。


- gfx1030 NVFP4 Index32のfull-tile境界除去＋実K定数化を、tile／dot／scale順を保つN0候補として固定比較した。
  M1024 K5120 N17408は`7760.042→8271.544 us`、K17408 N5120は`8274.326→8529.066 us`で退行。
  VGPRが86→104、active blocksが5→4へ悪化したため非採用。tiny FP64 oracle、実shape bit一致、
  M219の既存body維持はPASSしたが、同じ候補の追加探索は行わない。


- r15のgfx1030 FP8 ID71 full-tile候補は、実測M1024 K6144/N5120とK5120/N10240だけへ接続した。
  共有LDSを1組にし、VGPR52／LDS8704／active blocks 7を維持。production G1で全bit／oracle／repeat／cleanupをPASS。
  長文はr14 Graph-onとtoken一致、prefill `296.763→299.134 tok/s`、TTFT `31820.643→31565.963 ms`。
  decodeは`14.884→14.793 tok/s`であり、最終性能条件は未達のままである。
- r15ではgfx1201の既存FP8 Lt prepared planをrank変更なしでGraphへ接続し、代表M1K6144N5120の
  7入力／1,000 replay G1と、短文17/17の全生成token一致をPASSした。
  短文decode `19.503→19.508 tok/s`、TPOT `51.273→51.260 ms`で速度差は確認できない。
  長文off/onも全token／logical kernel監査が一致し、TPOT `54.993→53.856 ms`、decode
  `18.184→18.568 tok/s`（約2.1%改善）、HIP-only／cleanup 0を確認した。
  既存ID72を含む探索条件の比較であり、ID72のN2採用判断待ちは維持する。

- r17ではgfx1030 ID82のM1 K5120/N17408、K6144/N5120、K5120/N10240を独立symbolへ限定接続した。
  rolled loopと定数shapeで境界判定を省き、演算順はN0で維持。production G1の3形状と
  非一致K64/N32は全bit／独立oracle／repeat／finite／cleanupと公開dispatch symbol確認をPASS。
  専用kernelのVGPR170／active blocks 2は旧bodyへ波及しない。両target release build、Rust関連7 testをPASS。
  私有probeの128 MiB共通cache flush条件では追加2形状のkernel時間が20.87%／13.94%短縮したが、
  production G1の一部はMADが大きいため、その速度比を確定値にはしない。
  実モデル9435/128（1 warm＋3 measured）はr15と全token一致、HIP-only／nonfinite 0／cleanup 0をPASS。
  decode `14.793→15.224 tok/s`（約2.9%改善）、TPOT `67.602→65.686 ms`、
  prefill `299.134→300.074 tok/s`。最終性能条件は未達である。

- 開始処理の一時計測（gfx1201、17/17、1 warm＋3 measured）では、内部のgraph変換・検証が
  中央値9.478 ms、request buffersが2.594 ms、state生成が2.239 ms、後半のplan／queue／検証が8.706 ms。
  callerのgraph cloneは2.771 ms。outer setup中央値33.106 msで従来の約25 msより揺れがあり、
  各区間中央値を単純合計したり、以前の計測へ割合を適用したりしない。VMM mapはprefill側であり、
  request setupの支配要因ではない。次の調査対象はhost graph処理の重複で、変更の採用判断は未実施。

- r18/r19では無効な3種類のgraph rewriteの深いcloneを省き、WeightLoadPlanをresident/core間で
  不変共有した。要求別のgraph検証・状態・queueは維持。関連13 host testと両target release buildをPASS。
  r19の短文17/17（1 warm＋3 measured）は両targetで旧版と全token一致、HIP-only／nonfinite 0／cleanup 0。
  V620はsetup 23.431 ms、TTFT 234.249 ms、decode 16.186 tok/s。R9700はsetup 22.027 ms、
  TTFT 236.314 ms、decode 19.560 tok/s。R9700のr15 TTFT 241.619 msから約5.3 ms短縮したが、
  plan共有単独の追加効果はばらつき内。r18 V620は別buildと重なったためhost性能判断から除外した。
  長文の速度条件とR9700短文の不足は残り、追加のhost処理削減だけで完了とはしない。

- gfx1201 ID64の短文M17 K5120/N17408で、weight-scale乗算と累積加算を明示FMAへまとめる
  固定候補を測定した。geometry／K順／BF16出力丸めは維持し、中間丸め削減のN1として評価した。
  tiny M17K48N37と実shapeは全BF16 bit一致、FP64 sampled oracle／repeat／finite／cleanupをPASSしたが、
  共通cache条件の中央値 `928.951→1117.688 us`（約20.3%遅い）で非採用。VGPR90／LDS7684／scratch 0。
  productionへ接続せず、同候補の追加探索は行わない。

- gfx1030 Index32の2箇所の同期をLDS専用waitcnt＋barrierへ置換する候補は、tiny M32/33/65
  K48/N131とM1024 wide/downで全bit／数値oracle／repeat／finiteをPASSし、VGPR86／LDS6144／active 5を維持。
  固定4-copy比較はwide `9113.971→9065.250 us`、down `9991.584→9975.616 us`だが、
  差は0.54%／0.16%に留まり、raw/MADの記録がなく有意性は未確認。今回は本番接続せず非採用とした。
  初回probeは無効指定でも大規模CPU oracleを計算する不具合で中断し、tinyのみ全出力、
  実shapeは64点FP64 oracleへ修正したr2のexit 0結果を採用した。中断logは保持しGPU PASSに数えない。

- r19 R9700長文Graph-onのkernel trace（0 warm＋1 measured）はHIP-only／token一致／cleanup 0。
  191,336 trace行からruntime copy/fill 790行を除いた190,546行がauditと一致した。
  Dispatch_Id順の区切りでは127 decodeが各1341 dispatch、attention stage1/2各2032、argmax127。
  timestamp順には境界誤分類と負の境界差があるため、時刻は区間の観測値に限定する。
  decode raw event sum 6610.637 ms、union 5696.704 msで、差913.933 msを実concurrency／
  CPU・GPU idle／追加Graphで回収できる上限とは解釈しない。FP8 LtとNVFP4が主要なdevice演算。
  ID72のN2採用保留を維持し、このprofile時間は正式性能比較へ使わない。

- gfx1030 Index32のinput/output pointerへ`__restrict__`を付ける私有候補は、ELF symbolの
  全6284 byteを比較してcontrolと完全一致した（VGPR86／SGPR33／LDS6144／spill 0）。
  初回の99命令行はprologueだけで不十分だったため、symbol start＋sizeで全1147行を抽出して訂正した。
  命令・resource差がなく、この候補のGPU測定と本番接続は行わない。
  readonly input同士のaliasを禁止する新条件は導入しない。

- gfx1201 ID84 activation-LDS候補の初回比較は、HIP戻り値・cleanupの確認漏れ、
  repeat一致表示の実比較欠如、providerごとにまとめた測定順序が見つかったため、採用証拠に使わない。
  旧logは保持し、実PCI確認・反復出力比較・エラー伝播・交互測定へ修正したr4はexit 0。
  tiny全点FP64、実2shape全BF16一致、repeat／finite／cleanup 168/168をPASSした。
  通常中央値はwide `105.401→103.781 us`、down `107.166→105.786 us`で、改善を確認した。
  tinyは遅いため実2shapeだけへ接続し、r20 gfx1201 buildをPASS。実モデル確認は未実施。

- gfx1201 ID64のM17/K5120/N17408でpacked A/Wのload hintだけを通常loadへ変更した候補は、
  私有比較で `928.828→659.646 us`（MAD `3.020/9.301 us`）。4-copy、128共通prewarm、
  3 warm＋10交互measuredで、tinyと実shapeの全BF16一致、64点FP64、repeat／finite／cleanupをPASS。
  exact shapeへ接続したr20のproduction launcher対私有旧body G1もPASSした。
  同じload指定変更を既存down M17/K17408/N5120のsplit4へ適用する私有候補でも約33%短縮し、
  全BF16一致、FP64／repeat／cleanupをPASS。split4の分割・reduction順を維持してr21へ接続し、production G1もPASS。

- gfx1030 Index32のsingle-LDS raw-next-stage先読み候補はVGPR86／LDS6144／scratch 0／active 5を維持し、
  ISA上で次stage loadと現在stage dot4の重なりを確認した。tiny M32/33/65 K48/N131全点FP64、
  M1024 wide/down全BF16一致＋64点FP64、sentinel付きrepeat、finite／cleanupをPASS。
  4-copy／128共通prewarm／3 warm＋10交互measuredでwide `8229.996→7391.363 us`、
  down `8750.438→9465.679 us`。改善したexact M1024/K5120/N17408だけへr21で接続し、production G1もPASS。

- r21両target release buildはPASS。R9700の17/17（1 warm＋3 measured）は旧r19と全生成token・
  文章・auditが一致し、HIP-only／fallbackなし／nonfinite 0／cleanup 0。
  prefill `157.911 ms / 107.655 tok/s`、TTFT `179.119 ms`（MAD `1.175 ms`）、
  setup `21.504 ms`、decode `19.559 tok/s / TPOT 51.127 ms`、E2E `997.736 ms`。
  r19 TTFT `236.314 ms`から約57.2 ms短縮し、探索値ではllama短文TTFT `190.275 ms`を下回った。
  正式3 warm＋10 measuredは未実施であり、長文decode等のPhase 78完了条件は残る。
  r21 binary SHA-256はgfx1201 `621c5732e73f1d9e08757a91186fba5505a3a6ec74011a7a6e487bf798914080`、
  gfx1030 `aa18b46f2b8d08a414264696c4402ad2cd675af0a275f474b3a7e43831814b54`。

- r21長文9435/128（両target各1 warm＋3 measured）も旧版と全token・文章・audit一致、
  HIP-only／fallbackなし／nonfinite 0／cleanup 0。
  V620はprefill `30652.203 ms / 307.808 tok/s`、TTFT `30671.365 ms`、
  decode `15.266 tok/s / TPOT 65.504 ms`、E2E `38989.130 ms`。
  r17 prefill `300.074 tok/s`から約2.6%改善したが、prefill `340.80`／decode `16.86 tok/s`は未達。
  R9700はprefill `8197.300 ms / 1150.989 tok/s`、TTFT `8219.282 ms`、
  decode `18.719 tok/s / TPOT 53.422 ms`、E2E `15003.853 ms`。
  r15 decode `18.568 tok/s`から約0.8%改善したが、`21.07 tok/s`は未達。
  ID72 N2の採用保留、4行の正式3 warm＋10 measured、最終counter証拠は残る。
  gfx1030 ID71 full-tileのraw-next-stage先読みは私有sourceのcompile-onlyで打ち切った。
  control／candidateともcompile成功、LDS 8704 bytes／spill 0だが、VGPRは52→61へ増えた。
  次stage loadとcurrent dot2の重畳をISAから証明できず、解析時間が実装時間を超えたため、
  同候補の追加解析・GPU測定は行わない。productionはr21を維持する。

- gfx1201 FP8 decode M1/K5120/N10240のhipBLASLt rank7→8候補は、同じ32-result query／workspace 0、
  4独立weight copy／128共通prewarm／3 warm＋10交互measuredで再比較した。
  copy別中央値の平均はrank7 `0.124422 ms`、rank8 `0.124042 ms`で約0.3%差に留まり、
  過去のsingle-copy条件での約2.2倍差は再現しなかった。production policyはrank7を維持する。
  BF16／64点FP64／repeat／finiteの比較は通ったが、cleanupの戻り未確認をmain reviewで発見したため、
  cleanup PASSとは扱わず、このlogを採用用G1証拠には使わない。非採用判断のための追加GPU再測定は行わない。
  algorithm変更のN1採用用誤差bound証明も未実施であり、数値policyを拡張しない。

- gfx1030 ID82 exact M1/K5120/N17408の先読み数P2→P1をprivate compile-onlyで比較した。
  同一translation unit／compiler optionsでVGPR `170→90`、SGPR 26／LDS 544 bytes／spill 0。
  group数10→20により各laneのchunk順、dot順、reduction、scale、BF16 RNEは維持するN0候補である。
  GPU03の4独立copy／各provider各copy128共通prewarm／各copy3 warm＋10交互measuredでは、
  copy別中央値の平均がP2 `0.194811 ms`、P1 `0.211371 ms`で約8.5%遅く、P1は非採用。
  active blocksは2→5に増えたが速度へ転化しなかった。全17408列BF16一致、各copy64点FP64、
  実repeat／guard・finite／checked cleanupはPASS。productionのP2を維持し、P1追加測定はしない。
  同形状のactivation FP16-LDS共有も別のprivate compile-only比較でVGPR `170→144`、SGPR 26／spill 0。
  static LUT 544 bytesにdynamic activation LDS 10240 bytesを追加し、合計10784 bytesを要求する。
  GPU03 logではactive blocks 2→3、全列BF16／各copy64点FP64／実repeat／guard・finite／checked cleanupはPASS。
  ただしprobe binaryのpre/post SHA不一致が報告され、実行binaryとの対応付けは未確定である。
  このlogは採用用G1証拠に使わず、確定sourceに対するcorrectness PASSとは扱わない。
  同じ4-copy計測でcopy別中央値の平均はP2 `0.194941 ms`、共有案 `0.196861 ms`で約1.0%遅く、非採用。
  論理load要求削減は実DRAM転送削減の証拠にはせず、両案を合成した追加探索は行わない。
  このdecode微調整単位を閉じ、r21 P2を維持して既存計画の量子化共有へ進む。

- gfx1201 ID84のweight-only prefetch案は追加候補なし。既存r21 ISAの17個の64-bit loadは
  activation LDS初期化1個とweight 4列×P4の16個であり、未使用activation再loadは既にDCE済み。
  初回のactivation再load判定はaddress registerのdef-use追跡で訂正した。追加compile／GPU測定は行わない。

- 次の実装単位は既存項目5のFP8 GDN qkv/z activation共有をdecode M1へ限定したものとする。
  exact K5120／N10240,6144、同じBF16 input view／OCP E4M3FNの2-memberだけを対象とし、
  既存quantizerを一度呼び、同じbytes／row scaleを既存の2つのmatmulへ渡す。
  weight scale、provider／algorithm、各matmulの算術順序は変えないN0最適化である。
  受入範囲はrequest-owned workspace／既存completion・Graph契約を維持し、host contract、両targetの
  実GPU数値比較・repeat・cleanup、実モデルtoken一致と速度比較を既存の検証方針に従って確認すること。
  48組でquantizer起動を1組あたり2→1へ減らす。prefill／QKV triple／別shapeには拡張しない。
  速度改善が未確認の間はopt-in候補に留め、Phase 78の最終性能基準を緩めない。

- r23でFP8 GDN共有をnative／core／wrapper／HIP Graphへ接続した。
  `SLLM_QWEN38_FP8_GDN_PROJECTION_PACK2=1`が独立opt-inで、既存56 NVFP4 pairと同時に48 GDN pairをlowerする。
  M>1は従来の2 matmulへ分解し、M1のみ5124-byte workspaceを共有する。
  coreの関連18 test（1 ignored）と全体554 test（20 ignored）、native host test、両target release buildをPASS。
  mainの統合確認でNVFP4 grid監査値の退行を修正し、追加探索を止めてその修正へ検証を絞った。
  `.inc`単独変更をCargoが見逃す依存登録も修正し、修正後のarchiveを固定してG1を実行した。
  両targetでFP8共有3 dispatch／個別4 dispatch、member別weight・scaleの独立期待値、全出力一致、
  入力変更後比較、単一packの3-node Graph capture／replay、repeat、cleanup 0をPASSした。
  既存NVFP4 pairとV620 ID73の同じGPU testもPASS、実行前後のsource／archive／binary SHAは不変。
  r23 source manifest SHA-256は`75c0177bb01da93e1efaaa4a6944eebccd7d89840c0f5f4881573c6aa518fe53`。
  archive SHA-256はgfx1030 `402a213f60731a59b453af6985b1f4551f9103db6fd974c94cc1950ac5551d35`、
  gfx1201 `50da39102fbdd5093c8a173fa4d65d37e7e91d7a12bd148cfa2a67a743d99c71`。
  実モデルの短文／長文比較は下記のとおり完了した。Phase 78最終条件は未達である。

- r23の17/17と9435/128を両target各1 warm＋3 measuredで比較した。
  全16 runでbaselineとのtoken／文章／停止理由一致、HIP-only、nonfinite 0、request／resident cleanup 0を確認した。
  baselineはV620短文だけr19、それ以外はr21。model／fixtureは一致し、r23 binary SHAも保存identityと一致する。
  短文decodeはV620 `16.186→16.417 tok/s`、R9700 `19.559→19.719 tok/s`、
  TTFTはそれぞれ`232.206／180.982 ms`。長文decodeはV620 `15.266→15.402 tok/s`、
  R9700 `18.719→18.828 tok/s`、TPOTは`64.925／53.113 ms`へ改善した。
  長文prefillは`307.530／1145.812 tok/s`で、r21との差は両測定のMAD範囲内である。
  短文では768、長文では6096のsubmission／kernel削減、Graph capture nodeは1176→1128を全runで確認した。
  V620長文の物理VRAM sampled peakは30,172,577,792 bytes、GTT範囲17,100,800〜23,449,600 bytesで旧測定と同等。
  device-wide GTT値はdriver baselineを含むため、この観測だけを厳密なspill 0証拠とは扱わない。
  証拠は`.local-artifacts/phase78-resume/r23-fp8-gdn-{gfx1030,gfx1201}-{short,long}.*`と
  `r23-{gfx1030,gfx1201}-{short,long}-comparison.json`。N0共有をopt-inで維持する。
  V620 prefillと両target decodeの目標、最終4行3 warm＋10 measured、最終profile／counter、ID72のN2判断は未解決。
  同じr23 binaryの9435/128を各0 warm＋1 measuredでrocprof kernel traceへ記録し、通常測定との
  全token／文章／停止理由／audit一致とcleanup 0を確認した（`r23-profile-validation.json`）。
  これは次の最適化対象を選ぶ診断であり、正式性能比較や最終hardware counter証拠の代用ではない。
  Dispatch_Id順で10 prefill chunk／127 decode transitionへ分割した。runtime copyを除いたtrace数は
  V620 179914／R9700 184450でauditと一致し、decodeは両targetとも1 transitionあたり1293 kernelである。
  V620のdecode event時間はFP8約3301 ms、NVFP4約2740 msで合計約80.5%、
  R9700もFP8 Lt約2916 ms、NVFP4約2050 msで合計約78.5%を占める。
  V620 prefillのFP8／NVFP4合計も約82.3%であり、残差対策は行列演算本体を中心に再計画する。
  event sum／interval unionの差は真のidleや回収可能なGraph待ち時間とはみなさない。

- V620 ID62の通常load化は既に実装・測定済みである。r21 wide raw pipelineもdown Index32も通常loadを使う。
  過去のM1024 wide／down A/Bはそれぞれ約3.90%／5.65%改善を記録しており、同じ変更を新候補として再測定しない。

- 次の限定候補はgfx1030 NVFP4 decode ID84の実2形状で、既存P2／activation-LDS bodyへ
  M1/K5120/N17408またはM1/K17408/N5120を定数として渡すN0変更とする。
  loop内の境界判定・address計算削減だけを狙い、先読み数、load指定、dot4／FMA／reduction／丸め順は維持する。
  まず同一translation unitのruntime引数版と定数版をprivate compileで比較する。
  実行へ進む場合は既存方針に従い、両形状の独立数値期待値・全出力・repeat・cleanupと交互速度比較を確認し、
  改善した形状だけ本番へ接続する。過去に非採用としたprefill Index32の定数化とは別のdecode bodyである。
  独立候補として、既存ID82 rolled tuple bodyを未接続のGDN z M1/K5120/N6144だけで比較する。
  48回/tokenの現行generic bodyを対象に、既存P2・LUT・dot／reduction順を維持したN0定数化である。
  4独立weight copy、各provider共通prewarm、3 warm＋10交互measured、全BF16一致、
  境界を含む各copy64点の独立FP64期待値、repeat／guard／finite／cleanupを比較する。
  既存のK5120/N17408に対するP1・activation共有の再試行ではない。
  prefillは現行ID71 full-tileの実2形状だけで、協調loadの割当を隣接8-byteへまとめるN0案をprivate compileする。
  LDSの最終配置・変換・dot順は維持し、8-byte非整列pointerは既存bodyへ戻す。
  過去のraw-next-stage先読みとは独立したload幅の変更であり、追加のtile探索へ広げない。

- r24 GDN zのprivate比較はgeneric `0.090251 ms`→tuple `0.081091 ms`（4-copy中央値平均）。
  VGPR57→170／active blocks8→2でも約10.1%時間短縮し、全出力・FP64・repeat／guard／finite／cleanupをPASS。
  専用symbolとlauncher、shape別監査metadataを接続し、native host test、両release build、
  固定gfx1030 archiveのGDN共有／個別matmul／Graph replay／既存NVFP4 G1をPASSした。
  source manifest SHAは`65a600a23cb5be3bc52f1ddf490f6234364ef5b7744b85a9dc7137ce843ce4c7`、
  gfx1030 binary SHAは`1cd3c489b57b800799aad504853d696349f4691309ab1d2af02a9c79fdeb1be9`。
  短文／長文各1 warm＋3 measuredの全8 runでr23と全token／文章／停止理由／audit一致、HIP-only／cleanup 0。
  短文decodeは`16.417→16.450 tok/s`、TTFT `231.537 ms`。
  長文decodeは`15.402→15.412 tok/s`、TPOT `64.884 ms`で、decode時間差5.266 msはr24 MAD10.333 ms内。
  prefillは`308.308 tok/s`、TTFT `30625.455 ms`で、こちらもr23との差は旧MAD範囲内。
  whole-model速度改善は未確定としてID82 opt-in候補に留め、最終採用・Phase 78完了とは扱わない。
  証拠は`r24-fp8-z-*`および`r24-gfx1030-{short,long}-comparison.json`。
  NVFP4 decode定数化はcompile-onlyでVGPR63→51／static LDS1056／spill0、両実形状のGPU比較は準備中。
  FP8 prefill load64はVGPR52／LDS8704／spill0を維持し、aligned pathで2本の64-bit loadを確認したが、
  fallbackを含むsymbol sizeは23024→36076 bytesへ増えた。GPU数値／速度確認前に採用しない。

- r24 NVFP4 decode定数化はGPU03で両形状とも数値・全出力・repeat／guard／finite／cleanup114/114をPASSしたが非採用。
  4-copy中央値平均はwide `127.8405→128.11025 us`、down `131.3305→131.3705 us`で、copy間で一貫した改善がない。
  VGPR63→51でもactive blocksはwide8／down2のままである。runtime引数版を維持し、追加の定数化探索を行わない。
  harnessの列方向データ周期性、失敗時vector参照、printf型を実行前に修正したr2だけを証拠とする。
  実行前後SHA不変とrc0を`r24-nvfp4-consttuple-gpu03-r2.execution.json`へ記録した。

- r24 FP8 prefill load64の修正版r2 harnessはGPU03で両形状をPASSした。
  alignedは4独立copyの全出力・repeat・FP64 tile境界63/64/65、A/W非整列offset1/2/3/4/7は
  各copy0で既存fallbackとの全出力・期待値を比較し、guard／finite／checked cleanupも確認した。
  初版harnessの未確認cleanupはGPU実行前に修正し、初版binaryは実行していない。
  4-copy中央値平均はK6144/N5120 `3826.618→3569.996 us`、K5120/N10240 `6134.939→5683.695 us`。
  128共通prewarm、3 warm＋10交互measuredでcopyを巡回した。明示的cache flushは行っていないため、
  このprivate条件の改善を実モデル改善の証拠には置き換えない。
  r25としてこの2形状のaligned pathだけへ接続した。既存LDS配置・変換・dot順を維持するN0である。
  両release build、本番matmul TUを直接含む同じG1の全比較をPASSした（archive再リンク試験とは区別する）。
  source manifestは`12d409672170b2d3bbfa5db90a8afaf4d879a7b62463972796f668c3553514c3`、
  gfx1030 binaryは`4804159b3877790183193c909bac5400b01c006ed65535aa90c45059c65120e7`、
  archiveは`f30b4921c755f2e3297ab069cff1dc696c355b0b933a0f8afba8c35a26c26aa7`。
  固定r25 binaryのV620長文9435/128（1 warm＋3 measured）は全4 runでr24とtoken／文章／停止理由／audit一致、
  HIP-only／nonfinite0／request・resident cleanup0。
  prefill `30602.534→30229.262 ms / 308.308→312.115 tok/s`、TTFT `30252.204 ms`、
  E2E `38876.032→38475.186 ms`と改善した。decode `15.445 tok/s / TPOT64.748 ms`。
  prefill／decodeの目標と最終4行3 warm＋10 measuredは未達・未実施のままである。
  rawは`r25-fp8-load64-*`、比較は`r25-gfx1030-long-comparison.json`を参照する。

- 再開後の次候補はNVFP4 decodeのsigned-byte再構成である（private r26）。
  既存の6 permを2 permと整数演算へ置換し、dot・scale・FMA・reduction順を維持するN0候補として、
  gfx1030の現行P2 activation共有とgfx1201の現行P4 activation共有を対照にする。
  対象はM1/K5120/N17408およびM1/K17408/N5120。まずhost全符号組合せと両targetのcompile／ISAを確認する。
  GPUへ進む場合の受入条件は既存の全出力一致、独立数値期待値、repeat、guard、finite、checked cleanupを維持する。
  命令数・VGPRは判断材料であり新たなhard gateではない。性能改善は未確認、本番r25は変更しない。

- private r26 signedpackは両targetのcompileと実GPU比較を完了し、性能改善がないため非採用とした。
  host 65,536組合せは一致。gfx1030はVGPR63維持、gfx1201は91→92、両方spill0だが、
  perm削減をmul/sub/xor/and増加が伴う。比較はV620の本番constant LUT、R9700の本番activation共有へ揃えた。
  各2形状×4copyで全出力一致、独立FP64期待値、両providerのrepeat、guard／finite、checked cleanup114/114をPASS。
  128 prewarm、3 warm＋10交互measured、128MiB untimed flushの4-copy中央値平均は
  V620 wide `128.6185→131.03875 us`、down `132.4085→134.17875 us`、
  R9700 wide `106.009→106.569 us`、down `106.389→106.839 us`である。
  両実行rc0、source／binary SHA実行前後不変。rawは`r26-main-signedpack-gpu{03,07}`、
  集計は`r26-main-signedpack-gpu-summary.json`。本番r25を維持し、同じ候補の追加sweepは行わない。

- r25固定binaryで両targetの9435/128全モデルreadcounterを0 warm＋1 measuredで取得した。
  R9700は計測中だけprofile_standard、finallyでauto復元。V620はautoを維持した。
  両方rc0、全token／文章／停止理由／audit一致、HIP-only／nonfinite0／request・resident cleanup0。
  counterとkernel traceのDispatch_Id／Agent_Id／Kernel_Name集合は完全一致し、
  V620 180701行−runtime copy/fill787＝audit179914、R9700 185240−790＝184450で整合した。
  prefill10 chunk、decode127 transition×1293 kernelを確認し、4 binsの欠落・重複・nonfinite・負値は0。
  GL2C/EA read-request bytesはV620 prefill `4729892343424`、decode `2750205315968`、
  R9700 prefill `3019813036704`、decode `2768210160896`。
  decodeあたり各約21.655／21.797 GBだが、counter計測によるcache／実行への影響を含み得る。
  物理DRAM bytesや通常性能の帯域とは呼ばず、今回の計測時間を性能比較へ使わない。
  rawは`r25-{gfx1030,gfx1201}-long-readcounter*`、集計は`r25-counter-analysis-r2-{v,r}.json`、
  coverageは`r25-counter-coverage-validation.json`、数値確認は`r25-counter-model-validation{,-v}.json`。
  最終性能条件は未達であり、最終candidateが変わる場合はcounter証拠の適用範囲も確認する。

- FP8の単一family FP16 resident cache案はread-only feasibility調査で保留とした。
  既存ID86 on/offではstaging36.395 msをゼロとしてもconsumer777.696 msがID71の711.526 msより遅い。
  現行32GiBの観測peakから約4.187GB余裕、GDN z/out各familyの追加cacheは約3.020GBで容量上は候補だが、
  この変換削減だけでは性能改善の根拠にならない。実装・GPU再測定は行わない。
  QKV activation共有の追加効果もGDN実測比例の推定では1%未満であり、上限の証明ではない。
  次の調査候補はNVFP4 scaleだけのexact FP16事前展開で、重複履歴と追加容量の確認は未完了。

## Phase 79: 実用closeout

- artifactのstatic FP8 KVをappend、full attention、context growthへ直接接続し、FP16 mirrorを作らない。
- target-only逐次decodeが安定した後で、BF16 MTP companionを追加する。width 1〜3のdraft、逐次accept/reject、
  rejection replay、target-only同値、acceptance率とwall throughputを分離する。MTPが遅い場合はtarget-onlyをrollbackとして残す。
- 9,435-token実入力、128-token出力、短い対話、SSE、cancel/recovery、model unloadをsingle requestで確認する。
- 32 GB級deviceでmodel、MTP、KV、workspaceを収め、GTT spillとCPU/backend fallbackを許さない。
- visionはこのcloseoutをblockしない。text target達成後の独立機能項目とする。

## Phase 80: 他精度の単一要求最適化

Phase 79完了後にだけ、次の順で残件を閉じる。

1. MXFP8 W8A8 decode。ここで得たMXFP8 activation decodeを後続MXFP4 W4A8へ再利用する。
2. MXFP6 W6A6 decode。MXFP8のtile/reduction骨格を使い、E3M2 ingressだけを独立評価する。
3. MXFP4 W4A8 prefill/decode。weightはMXFP4 block32/E8M0、activationはMXFP8 E4M3 block32/E8M0とする。
4. NVFP4 W4A16 decode残差と、必要なら既存prefill providerの追加改善。

一般的なFP8 artifact互換は保留を維持する。Phase 76〜79のexact Qwen3.8 recipe対応を汎用化する作業はPhase 80へ
自動的に含めない。

## Phase 81: NVFP4 batching

Phase 80完了後に開始する。最初はQwen3.8-27B NVFP4 W4A4へ限定する。

1. 同一decode stepのB=`2/4/8`でactivation pack、weight tile、scale loadをrequest間共有し、単一要求TPOTとaggregate throughputを測る。
2. Phase 26のhost planningを再利用し、GPU B>1 executionへ接続する。単一要求providerを暗黙にB>1へ流用しない。
3. decode-only batchingを成立させた後、prefill/decode混在、continuous admission、cancellation、KV ownershipへ進む。
4. fairness、p50/p99 latency、aggregate tok/s、resident/request workspace、OOM admissionを別指標として記録する。

tensor parallel、multi-GPU、RDMAはPhase 81へ含めず、batchingと通信最適化を同時に導入しない。

## 共通停止・再計画条件

- exact artifact recipeまたはmodel revisionが変わった場合は、既存bytesを新recipeへ読み替えずlockとinventoryを再確認する。
- profileで想定と異なるconsumerが支配的なら、Phase内の候補順だけをwall寄与順へ変更し、別precisionやbatchingへscopeを広げない。
- 二度不採用になった同じcandidate、実装時間を超えるreview、機能進捗停止、見積り1.5倍超では追加探索を止めて再計画する。
- GPU PASSはexact target、数値oracle、fallback 0、GTT spill 0、cleanup成功を必要とする。

[メイン計画](../../../../main-plan.md)
