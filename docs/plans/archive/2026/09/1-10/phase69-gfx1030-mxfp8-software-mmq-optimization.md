# Phase 69: gfx1030 MXFP8 software-MMQ次段最適化

状態: `完了・ID41 vector32 ingressを既存Phase 67 scopeへ採用`

## 完了記録（2026-09-02）

- P69-AでID27を再取得し、Code Object resourceとrocprofv3 counterを固定した。ID41はID27と同じLDS 8,704 byte、
  VGPR 46、spill 0を維持し、VALUInsts平均を19.44%削減した。
- P69-Bのregister-scale ID40とP69-B/C combined ID42はVGPR 53／54とshape別非単調性によりbenchmark-onlyとした。
- P69-Cの32-bit E4 value ingress ID41はN0でBF16 digestを維持し、28-case operator oracleをPASSした。
- P69-Dはprofileが二重bufferを支持せず、P69-EはN0候補がfull-modelを改善したため実装しなかった。
- 同一最終binary、3 warmup＋10 measuredのQwen3.5-4B primaryは512 inputで
  `205.0009 -> 254.4461 tok/s`（+24.12%）、2,048 inputで`204.2416 -> 249.3441 tok/s`（+22.08%）。
  生成token、dispatch件数、resident／peak VRAM、HIP-only、fallback、cleanupは不変だった。
- ID41をexact `gfx1030`の既存Phase 67 production scopeへ採用した。旧ID27へは
  `SLLM_MXFP8_PREFILL_FORCE_MMQ_GFX1030_PHASE69=control`、row8へは
  `SLLM_MXFP8_PREFILL_FORCE_ROW8=1`で戻せる。

## 目的

Phase 67で限定採用したID27 col8と、Phase 68で採用した内部MX value-plane fast pathを基準に、native FP8
matrix命令を持たないexact `gfx1030`のOCP MXFP8 prefillをさらに最適化する。永続BF16／FP32 weight plane、
FP32 attention／KV、request間activation cacheは追加せず、weight／activationをpacked MXFP8のまま保持する。

このPhaseは次の3段階のうち第1段階だけを実行する。

1. Phase 69: exact `gfx1030`のMXFP8 E4M3 software-MMQを最適化する。
2. 後続候補: MXFP6 E3M2をtile load時だけE4M3へexact変換し、Phase 69で整理するMMQ骨格を再利用する。
3. 後続候補: NVFP4はblock 16、E4M3 block scale、FP32 tensor scale、W4A16／W4A4を保つ独立specializationとする。

MXFP6／NVFP4の実装と採否はPhase 69のscope外であり、Phase 69の証拠を得た後に別Phaseとして詳細化する。

## 固定baseline

- hardware／software: canonical Radeon Pro V620、exact `gfx1030`、ROCm 7.14.0、Code Object V6、wave32。
- format: OCP MXFP8 E4M3 W8A8、block 32、E8M0 scale、FP32 accumulation、BF16 RNE output。
- runtime: Qwen3.5-4B MXFP8、FP16 KV、direct pretokenized input、最大4 output、greedy、ignore EOS。
- provider: Phase 67のID27 col8 scoped defaultとPhase 68の内部MX value-plane fast path。
- selector scope: `M>=128, K>=2048, K%32=0`かつ`2560<=N<=16384`または`M>=512 && N==1024`。
- rollback: `SLLM_MXFP8_PREFILL_FORCE_ROW8=1`。Phase 69候補にはcandidate単位の明示overrideも設ける。
- Phase 68最終CLI identity: `sha256:af687e0d4562bbf478be537bf6f9582d48a4ef9d2a62a477714063c81ed94c7c`。
- Phase 68 full-model中央値: input 512が`213.0431 tok/s`、input 2,048が`213.0759 tok/s`。
- Phase 67 ID27 resource: LDS `8,704` byte、SGPR `29`、VGPR `46`、spill 0、wave32。

Phase開始時に同じsource／artifact／入力でcontrolを再取得し、古い絶対値だけをcandidateの比較対象にはしない。

## 受入条件

- selector keyはexact target、format、layout、M/N/K、alignment、resource条件だけとする。model名、layer名、prompt、
  token ID、計測結果をselectorへ含めない。
- weight／activationのvalue byteとE8M0 scale byte、scale 255によるNaN block、signed zero、subnormal、Inf saturation、
  FP32 accumulation、BF16 RNEの契約を維持する。
- まず実数式、項、演算順、丸めstageを維持するN0候補を評価する。N1は数値変更台帳の解析条件を満たす場合だけ通常採用候補にできる。
  N2は性能・誤差・出力差を提示する研究候補に留め、ユーザーの明示判断なしにproduction採用しない。N3は棄却する。
- candidateは独立oracle、特殊値、boundary、HIP-only、fallback false、repeat determinism、resource、cleanupを確認するまで
  benchmark-onlyとする。
- 固定の性能向上率をhard gateにしない。同一binaryのpaired測定で雑音を超える安定した改善があり、対象full-model行を
  退行させない候補だけをscoped defaultへ採用する。
- persistentな展開weight、FP32 attention／KV、cross-request cache、public ABI、KV default、samplingは変更しない。

## 候補と実行順

### P69-A: fresh baselineと時間分解

- ID27のactivation quantize、value ingress／E4 decode、scale ingress、inner product、reduction、barrierを分離して計測する。
- kernel resource、occupancy、LDS access、global load、VALU、barrier stallを取得し、以降の候補順を確定する。
- Phase 67で棄却したID38／39 col16／col32、weight direct-loadと、Phase 68で棄却したpre-scaled weight／wave ballotは
  controlとして記録するだけにし、同じ実装を再試行しない。

### P69-B: K32 scaleのregister化（N0第一候補）

- K tile内をblock 32単位のcompile-time stepへ分け、activation scale 1個と各output列のweight scaleをregisterへhoistする。
- inner loop内の動的block index、除算、反復LDS scale readを削減する。
- 各laneのvalue順、scale適用位置、FP32 term／accumulator順、BF16 RNE stageはID27と一致させる。

### P69-C: vectorized E4 value ingress（N0第二候補）

- 整列したvalue planeを32／64／128-bit単位で読み、複数E4M3 codeのunpackとnormal FP32 bit構築をまとめる。
- zero／subnormal／内部quantizerのscale 255契約はPhase 68のrare pathへ送る。非整列tailは既存scalar pathを維持する。
- scalar load比でglobal transaction、VALU、VGPR、occupancy、operator時間を比較し、load幅は測定で選ぶ。

### P69-D: stagingとbarrier schedule（N0第三候補）

- ID27 col8のoutput幅を固定したまま、producer割当て、LDS layout、load/store順、barrier回数を変更する候補を評価する。
- 二重bufferはfresh profileがmemory latencyまたはbarrier stallを主要因と示す場合だけbounded candidateとして作る。
- LDS／VGPR増加によるoccupancy低下を必ず記録し、col16／col32を形だけ変えずに再導入しない。

### P69-E: block contribution後scale（N1/N2研究候補）

- N0候補後もscale multiplyが支配的な場合だけ、block 32のunscaled contributionを先に集約してscaleを一度適用する案を解析する。
- real-number式が同じでもFP32の演算順と丸めstageが変わるため、誤差boundを先にN1／N2へ分類する。
- N2ならoperator／full-model性能と最初のlogit／token差まで提示し、Phase 69内ではproduction採用しない。

integer mantissa/exponent bucket、DP4A化、永続的な別weight表現はこのPhaseへ無条件に追加しない。P69-Aでdecode／dotが
明確な残差となり、上のbounded候補で進展しない場合に次Phase候補として記録する。

## 検証matrix

### Host／operator

- admission境界: M=`127/128/129/511/512/513/2047/2048/2049`、K=`31/32/33`、N=`1023/1024/1025/2559/2560/2561`。
- 実shape: K/N=`2560/1024`、`4096/4096`、`9216/2560`、`4096/8192`、`4096/9216`とPhase 67で採用済みのwide/down shape。
- value: `+0/-0`、normal、全subnormal、最大有限、Inf saturation、standalone E4 NaN、scale `0/1/118/127/134/254/255`。
- 独立host oracle、control/candidate BF16 digest、repeat digest、provider identity、HIP-only、fallback、cleanupを記録する。
- resource: code object、wave size、SGPR、VGPR、LDS、spill、occupancyと主要profiler counterをcandidateごとに固定する。

### Full-model

- primaryは固定Qwen3.5-4B、input 512／2,048、FP16 KV、最大4 outputとする。
- draftは1 warmup＋3 measured、最終候補は3 warmup＋10 measuredとし、同一最終binaryでPhase 68 controlとpaired比較する。
- prefill token/s、prefill時間、E2E、生成token、resident／peak VRAM、provider dispatch数、fallback、cleanupを記録する。
- Qwen3.5-9Bはreviewed固定artifactを同一条件で再利用できる場合だけmodel共通性の補助確認に使い、4Bの採否をblockしない。
- primary外のGPU、KV形式、model、長context、API／WebUI全行の広域rerunをPhase 69の必須条件にしない。

## 採否と完了条件

- candidateは`shared`、`gfx1030 shape-scoped`、`benchmark-only`、`rejected`のいずれかへ分類する。
- default採用はexact `gfx1030`かつ測定済みshapeだけに限定し、scope外はPhase 68のID27またはrow8へ戻す。
- 速度改善候補がなくても、全候補の数値分類、性能、resource、棄却理由、rollbackを固定すればPhase 69は完了できる。
- 採用変更が数値出力へ影響する場合だけ数値変更台帳を更新し、target/toolchain対応範囲が変わる場合だけ互換性文書を更新する。
- 完了時にplanをarchiveへ移し、matching historyと追跡summaryを作成してmain planを同期する。

## 後続formatへの引継ぎ境界

Phase 69ではMMQ本体からcompile-timeのtile loader／block policy境界を分離し、MXFP8でno-regressionを確認する。
後続MXFP6はresident E3M2 packed valueを保持し、tile load時にnormalのsign／exponent／mantissaをbit変換し、subnormalを
固定mapしてE4M3へexact変換する。normalはsign bit 5をbit 7へ移し、exponent biasを`+4`、2-bit mantissaを1-bit左shiftする。
正のsubnormal codeは`0/1/2/3 -> 0x00/0x18/0x20/0x24`、負値はE4M3 sign bitを付ける。
3 packed byteから4 codeをまとめて生成し、元のblock 32／E8M0 scaleをそのまま使う。
whole-model E4／FP32展開は行わない。

NVFP4は共通のschedule／reduction骨格だけを再利用し、E2M1 value、block 16、E4M3 block scale、FP32 tensor scale、
W4A16／W4A4別のloader／scale specializationを持つ。MXFP8／MXFP6のscale policyへ偽装しない。

[全体計画](../../../../main-plan.md) /
[Phase 68 baseline](../../../../archive/2026/09/1-10/phase68-gfx1030-mxfp8-e4-scale-fast-path.md) /
[履歴](../../../../../history/2026/09/1-10/phase69-gfx1030-mxfp8-software-mmq-optimization.md) /
[追跡要約](../../../../../../ci/matrix/phase69-gfx1030-mxfp8-software-mmq-v1.json)
