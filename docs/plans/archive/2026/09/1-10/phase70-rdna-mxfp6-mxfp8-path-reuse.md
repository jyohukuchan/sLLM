# Phase 70: 両RDNA MXFP6のMXFP8実行骨格再利用

状態: `完了・P70-F ID45 packed-group N64限定採用／ID46 N128 benchmark-only`

## ユーザー決定と対象

2026-09-02のユーザー指示により、Phase 69後の次PhaseをPhase 70とし、exact `gfx1030`とexact `gfx1201`の
OCP MXFP6 E3M2 W6A6 prefillを、既存MXFP8経路の実行骨格をほぼ共有する形へ実装する。

対象はmodel weight／activationのMXFP6 W6A6、block 32、E8M0 scale、FP32 accumulation、BF16 RNE outputである。
resident weightとmatmul前段のactivationは現在のpacked E3M2（4 value／3 byte）を維持し、whole-modelまたは
request全体のE4M3／BF16／FP32展開は作らない。KV形式、KV default、decode M=1、sampling、model recipe、GGUF encoding、
public APIは変更しない。

## 目的と完了時の成果

- format固有処理をpacked E3M2の読み込み、3-byte→4-code unpack、E3M2→E4M3 exact変換、source strideへ限定する。
- `gfx1030`ではPhase 69 ID41のcol8 MMQ schedule、E4 decode、scale、FP32 accumulation、wave reduction、BF16 outputを再利用する。
- `gfx1201`ではPhase 63〜66のFP8 WMMA fragment、K32 scale適用、FP32 accumulator、N tile、BF16 outputを再利用する。
  packed 6-bit operandはrocWMMAへdirect-loadできないため、tile ingressだけでE4M3 byteへexact変換してLDSへstageする。
- MXFP8とMXFP6を同じprovider identityへ偽装せず、formatとresident layoutは別のまま、prepare時にtarget別の具体的な
  ingress／tile／inner-product specializationをfreezeする。
- 両targetでcandidateを実装・実行し、operator、resource、profile、固定Qwen3.5-4B full-modelから
  `shape-scoped default`、`benchmark-only`、`rejected`のいずれかへ分類する。固定の速度向上率はPhase完了条件にしない。

## 固定baseline

- software: ROCm 7.14.0、Code Object V6、現行`main`のPhase 69完了source。
- hardware: canonical Radeon Pro V620 exact `gfx1030`とRadeon AI PRO R9700 exact `gfx1201`、いずれもwave32。
- artifact: Qwen/Qwen3.5-4B MXFP6 GGUF
  `sha256:d0ff2e1de9d87dddddcde8f85ef305bbf21a06d5f7586d077ba1178580a0264e`、明示FP16 KV。
- 現行MXFP6 variant: decode ID20、baseline prefill ID21、row8 ID23、tiled16 ID25、MMQ col4／col8 ID28／29。
  Phase開始時に両targetの実dispatchとfresh 512／2,048-token baselineを取得し、古い短入力値だけで採否しない。
- 既知の参考値は、gfx1201の2,048-token MXFP6 current providerが1 warmup＋3 measuredで中央値
  `301.984 tok/s`、gfx1030／gfx1201の17-token col8が`109.88／131.91 tok/s`である。
- 共有経路の退行controlとして、同じ最終sourceからMXFP8も再buildし、gfx1030 ID41とgfx1201 ID36／37の
  selector、operator digest、resource、代表full-model行が意図せず変化していないことを確認する。

## 再利用する設計境界

### 共通format ingress

- `low_precision_block_codec.hpp`または同等のdevice-inline層へ、packed 24-bitから4個のE3M2 codeを取り出し、
  4個のOCP E4M3FN byteへexact変換するprimitiveを一つだけ定義する。
- E3M2 normalはsign bitをbit 5からbit 7へ移し、exponent biasを`+4`、2-bit mantissaを1-bit左shiftする。
  正のsubnormalは`0/1/2/3 -> 0x00/0x18/0x20/0x24`、負値はE4M3 sign bitを付ける。signed zeroを維持する。
- E3M2にはInf／NaN codeがない。入力Infは既存quantizerで最大有限へsaturateし、NaN blockはE8M0 scale `255`と
  zero value planeで表す現行契約を維持する。scale byteは変換せずblock 32のまま使う。
- 全64 E3M2 codeと全packed lane位置について、`decode_e3m2(code)`と`decode_e4m3fn(convert(code))`の
  FP32 bit一致をhost／deviceの独立oracleで固定する。

### gfx1030 software-MMQ

- Phase 69で分離したpacked-value ingressへMXFP6用`3 byte -> 4 E4M3 -> existing E4 FP32 decode` policyを追加する。
- row／column／K分解、col8 weight再利用、activation／weight scale、LDS配置、FP32 term順、wave reduction、BF16 RNEは
  ID41本体を共有する。activationとweightの両operandへ同じexact converterを使い、片側だけ旧scalar E3M2 decodeに残さない
  candidateをprimaryとする。
- 現行MXFP6 col8 ID29と、既存E3M2 direct decodeを同じscheduleへ接続したcontrolを残し、変換cost、VALU、load、LDS、
  occupancyを分離する。MXFP8 ID41のformat identityや既定selectorを変更しない。

### gfx1201 FP8 WMMA

- Phase 63〜66のWMMA bodyをoperand ingress policyでtemplate化し、MXFP8は現行direct byte ingress、MXFP6は
  packed E3M2からE4M3 byteへ変換するstaged ingressを選ぶ。
- MXFP6は3-byte packingのためID36／37のglobal direct-loadをそのまま使用できない。activation／weightのK32 tileだけを
  E4M3 byteとしてLDSへ置き、rocWMMAのFP8xFP8→FP32、block scale適用、accumulator、output mappingを共有する。
- 最初にN64 staged候補を作り、LDS／VGPR／occupancyに余裕がある場合だけN128を同じbodyの別tile parameterとして評価する。
  N128を作ることや勝たせることは完了条件にせず、resourceまたは性能で不利ならN64までで採否を確定する。
- MXFP8 direct-load instantiationは既存kernel symbol／selectorを維持する。共通化によるsource-level refactorでcodegenが変わる場合は、
  MXFP8 operatorとfull-model controlを再取得して意図しない退行を採用しない。

## 作業単位

### P70-A: fresh baselineとexact converter

1. 両targetでID20／21／23／25／28／29のselector、operator時間、resource、512／2,048 full-modelを取得する。
2. scalar code 64通り、4 laneのpacked 24-bit、block scale `0/1/118/127/134/254/255`を含むconverter oracleを追加する。
3. standalone converterとmatmul内converterの出力を一致させ、別consumer用の重複変換関数を作らない。

### P70-B: 共通operand-ingress境界

1. MMQとWMMAのhot bodyからresident stride／unpack／tile materializationをcompile-time policyへ分離する。
2. provider planへMXFP6のtarget別MMQ-via-E4／WMMA-via-E4 identityを追加し、MXFP8 providerとは別に監査可能にする。
3. prepare後の環境変更でvariantが変わらないこと、model名・layer番号・prompt・token・測定結果がselector keyへ入らないことを
   host testで固定する。

### P70-C: gfx1030 MXFP6 col8-via-E4

1. 3-byte groupを整列loadし、4 E4M3 byteへ変換して既存ID41 E4 decodeへ渡すcandidateを実装する。
2. control／candidateのFP32項順を一致させ、N0のBF16 digest一致をprimary条件としてoperator shape sweepを行う。
3. current col8、tiled16、row8とのcrossoverを測り、勝つshapeだけmodel非依存selector候補にする。

### P70-D: gfx1201 MXFP6 staged-WMMA

1. activation／weightのK32 tileをE4M3 LDS tileへ変換するN64 candidateを実装する。
2. E4M3 WMMA contributionへ既存E8M0 scale pairを適用し、現在のMXFP6 real-number式とBF16 output契約を維持する。
3. N64のprofileとresourceが支持する場合だけN128を追加し、ID25 tiled16、ID29 col8、N64／N128を同一runnerで比較する。

### P70-E: selector、full-model、closeout

1. exact target、format、layout、M/N/K、alignment、resourceだけからtarget別shape scopeを決める。
2. 同一最終binaryでcontrol／candidateをpaired測定し、生成token、provider dispatch、resident／peak、HIP-only、fallback、cleanupを
   比較する。候補が雑音を超えて勝たないshapeは既存providerへ戻す。
3. 採用／benchmark-only／棄却結果、rollback、数値分類、code object／binary identityを追跡summaryとmatching historyへ固定し、
   planをarchiveへ移してmain planを同期する。

### P70-F: gfx1201 packed-group ingressとN128 follow-up

2026-09-02の追加指示により、ID44とMXFP8 ID31／36／37の現行binaryをkernel traceで比較し、512-token主projection差を
N128→N64 `15.3%`、direct-load body→LDS staged body `49.2%`、packed E3M2抽出＋E4M3変換 `35.5%`へ分離した。
この結果をPhase 70の追加work unitとして次の順に実行する。

1. packed E3M2を1 valueごとに同じ3 byteから再読込するID44をcontrolとして保持し、4 value／3 byteを一度だけ読み、
   4個のexact E4M3FN byteをまとめてLDSへ書くN64 candidateを別kernel identityで追加する。
2. N64 candidateがoperatorで改善した場合は、同じpacked-group ingressをN128 output tileへinstantiationする。resident layout、
   block 32／E8M0 scale、FP32 accumulation、BF16 output、M=1 decode、scope外providerは変更しない。
3. exact `gfx1201`のM=`17/127/128/512/2048`、K=`2560`、N=`9216`をprimary operatorとし、ID44／N64／N128を
   1 warmup＋3 measuredで比較する。独立FP32 oracle、非有限位置、repeat determinism、actual kernel ID／symbolを必須とする。
4. resourceはLDS、SGPR、VGPR、spill/private、static WMMA／VALU／VMEMを比較する。derived counterが再び無効値を返す場合は
   採否根拠にせず、kernel timingとcode object resourceを使用する。
5. 固定Qwen3.5-4B MXFP6、FP16 KV、direct input 512／2,048、4 output、greedy、ignore EOSを同一最終binaryでpaired測定する。
   draftは1+3、採用候補は3+10とし、ID44より安定して速いshapeだけmodel非依存selectorへ採用する。
6. MXFP8 ID37の512／2,048代表行、host selector freeze、gfx1030／gfx942非選択、rollback、cleanupを確認し、結果を
   Phase 70のplan／history／summaryとmain planへ同期する。

P70-Fは真のpacked zero-copy WMMAを完了条件にしない。rocWMMA fragmentへpacked 6-bitを直接渡せない現行公開APIでは、
変換後のtile-local E4M3 LDS materializationを維持する。persistent E4M3 weight、request全体のE4 activation cache、
新しいquality recipe、別GPUの実機再測定は対象外とする。

P70-Fは完了した。ID45 N64は4 value／3 byte groupを一度だけ読み、4個のexact E4M3FN byteを32-bit単位でLDSへ書く。
ID44のscalar ingress、ID45、ID46 N128を同一最終operator runnerで比較し、主shapeのID45はID44比`1.535〜1.838倍`だった。
ID46はM=2,048 operatorだけID45比`1.061倍`だったが、M=17／127／128／512では`0.478〜0.623倍`へ退行し、
実モデル512／2,048-tokenでもID45に負けたためbenchmark-onlyとした。

## 数値分類

- packed E3M2→E4M3変換そのものは全E3M2値の実数値をexactに保つN0とする。
- gfx1030はID41と同じFP32 term／reduction順を保ち、controlとのBF16 digest一致を要求するN0候補とする。
- gfx1201はcurrent tiled16／MMQからFP8 WMMA treeへ変わるため、format変換がexactでもprovider全体はN1候補として扱う。
  各K32の項とscaleを欠落させず、独立FP32 oracle、非有限位置、repeat determinism、最大absolute／relative errorを記録する。
- gfx1201でcontrolとのlogit／token差が生じた場合は最初の差、top-1、KLD、perplexityを記録する。旧KV default用の`0.99`
  thresholdをMXFP6 W/A providerへ流用せず、MXFP6 quantization recipe自体の既知品質とprovider算術差を分離する。

## 検証matrix

### Host／operator

- target: exact `gfx1030`／`gfx1201`。`gfx942`とunknown targetはcandidate非選択をhost testし、gfx942はcompile-onlyとする。
- boundary: M=`1/17/127/128/129/511/512/513/2047/2048/2049`、K=`31/32/33/2048/2560/4096/9216`、
  N=`31/32/33/63/64/65/127/128/129/1023/1024/1025/2559/2560/2561`。
- production shape: Qwen3.5-4Bのwide、down、gate/up、output、N=1024、N=32。K%32!=0は従来どおりfail-closeする。
- value: signed zero、全E3M2 subnormal／normal／最大有限、入力Inf saturation、NaN block、E8M0最小／有限／255。
- evidence: independent FP32 oracle、BF16 digest、repeat digest、kernel ID／symbol、fallback、HIP-only、cleanup。
- resource: wave size、workgroup、LDS、SGPR、VGPR、spill/private、static WMMA命令、occupancy、VALU、MemUnit／LDS／barrier counter。

### Full-model

- primary: 固定Qwen3.5-4B MXFP6、direct pretokenized input 512／2,048、FP16 KV、最大4 output、greedy、ignore EOS。
- draftは1 warmup＋3 measured、最終候補は3 warmup＋10 measuredとし、全sample、median、MAD、prefill時間、E2Eを残す。
- 両GPUで同じmodel artifact、入力token file、最終sourceを使う。GPU間の絶対倍率は参考とし、採否は各GPU内のpaired比較で決める。
- Qwen3.5-9Bは固定MXFP6 artifactを再利用でき、primary完了後に容量と時間が許す場合だけmodel共通性の補助行とする。
- 共有sourceの退行controlとして、MXFP8 Qwen3.5-4Bの512／2,048代表行を両targetで再実行する。

## 採否と完了条件

- 両targetで少なくとも一つのMXFP6-via-E4 candidateを実装し、GPU operator oracleとfull-modelを実行する。
- candidateはtarget／shapeごとに`shared implementation`、`shape-scoped default`、`benchmark-only`、`rejected`へ分類する。
- production採用は同一binary paired測定で安定して勝ち、対象full-model行を退行させないscopeだけに限定する。
  改善率未達を理由にPhaseを未完了にはせず、遅い候補は既存ID25／29等を維持して結果を閉じる。
- MXFP8、decode、scope外shape、別targetの既存selectorはrollback先として残す。実行失敗後のsilent fallbackは追加しない。
- persistent E4M3／BF16／FP32 weight、request全体のE4M3 activation cache、FP32 attention／KV plane、NVFP4、MXFP4、
  model quality recipe変更はPhase 70へ含めない。

## 完了結果

- packed E3M2からE4M3FNへのexact変換を共通codecへ追加し、全64 code×4 packed laneをexact `gfx1030`／`gfx1201`の
  device oracleでPASSした。resident weight、activation carrier、block 32、E8M0 scaleは変更していない。
- exact `gfx1030`向けID43は既存col8 MMQのschedule／E4 decode／scale／FP32 reductionを共有した。operator出力は
  current ID29とproduction shape全行でdigest一致したが、固定Qwen3.5-4Bの512／2,048-token prefillがそれぞれ約22.7%／
  21.6%遅かったため、明示benchmark専用として残し既定selectorへ採用しなかった。
- exact `gfx1201`向けID44はMXFP8 N64 WMMA bodyをoperand-ingress policy化し、packed MXFP6 tileだけをLDS上のE4M3へ
  materializeした。P70-Fでは同じscopeへpacked 4-value ingressのID45を追加してID44を置換し、model非依存で既定採用した。
  明示rollbackとしてID44を選ぶ`SLLM_MXFP6_PREFILL_FORCE_PHASE70=gfx1201-n64`と、従来tiled16を選ぶ
  `SLLM_MXFP6_PREFILL_FORCE_TILED16=1`を追加した。
- gfx1201の固定Qwen3.5-4B 3 warmup＋10 measured paired比較では、512-token中央値が`307.588→1302.342 tok/s`
  （4.234倍）、2,048-token中央値が`299.929→1508.692 tok/s`（5.030倍）となった。生成token、3,008 dispatch、
  resident／peak VRAM、HIP-only、fallback 0、cleanup 0はcontrolと一致した。
- gfx1201 ID44はFP8 WMMA reduction treeへ変わるN1 providerとして、独立FP32 oracle、非有限位置、repeat determinismをPASSした。
  512／2,048 operator digestは旧providerと異なり得るが、full-modelのgreedy生成tokenは全反復で一致した。
- P70-Fの同一最終binary 3 warmup＋10 measuredでは、ID44→ID45の512-token中央値が
  `1276.494→2157.868 tok/s`（1.690倍）、2,048-token中央値が`1506.933→2423.308 tok/s`（1.608倍）となった。
  prefill時間は`40.84%／37.82%`、E2Eは`29.30%／33.68%`短縮し、全sampleの生成token、3,008 dispatch、
  resident `4,061,763,072` bytes、peak `4,400,391,680／5,261,350,400` bytes、HIP-only、fallback 0、cleanup 0が一致した。
- gfx1030／gfx1201のMXFP8 operatorと512／2,048-token代表行を同一最終sourceから再実行し、既存ID41／ID37の
  selectorと既知速度水準を維持した。P70-F後のgfx1201 MXFP8は512／2,048-token中央値`3869.523／3775.148 tok/s`だった。
  gfx942はcandidate非選択host testとrelease compile-onlyをPASSした。
- 最終code objectではID44／45／46がそれぞれLDS `6,912／6,912／9,216` bytes、SGPR `34／38／34`、
  VGPR `114／115／167`、spill/private 0、static WMMA `8／8／16`だった。ID46の高いVGPRとwide-tile退行を根拠に
  benchmark-onlyとした。rocprofv3の対象derived counterは当該環境で0を返したため、
  採否にはcode object resourceとstatic instruction分類を使用し、無効counterを性能根拠にしなかった。

## 対象外と後続

- NVFP4はE2M1、block 16、E4M3 block scale、FP32 tensor scale、W4A16／W4A4を持つため、Phase 70のMXFP6 adapterへ
  接続しない。Phase 70完了後の独立specialization候補として順序を維持する。
- gfx942、RDNA3、CPU、CUDA、新model architecture、KV量子化、attention kernel、multi-GPU、batchingは対象外である。
- external engineの再調査や第三者code importは開始条件にしない。既存sLLM codec、Phase 69 ingress境界、rocWMMA公開APIから実装する。

[全体計画](../../../../main-plan.md) /
[Phase 69保存済み計画](../../../../archive/2026/09/1-10/phase69-gfx1030-mxfp8-software-mmq-optimization.md) /
[Phase 66履歴](../../../../../history/2026/09/1-10/phase66-gfx1201-reusable-low-precision-attention-transfer.md) /
[Phase 69履歴](../../../../../history/2026/09/1-10/phase69-gfx1030-mxfp8-software-mmq-optimization.md) /
[Phase 70履歴](../../../../../history/2026/09/1-10/phase70-rdna-mxfp6-mxfp8-path-reuse.md) /
[Phase 70追跡要約](../../../../../../ci/matrix/phase70-rdna-mxfp6-mxfp8-path-reuse-v1.json)
