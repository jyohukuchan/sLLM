# Phase 75: gfx1030 MXFP8先行・MXFP6共通half2最適化

状態: `完了`

## ユーザー決定と目的

2026-09-03のユーザー指示により、Phase 74後の次PhaseをPhase 75とし、exact `gfx1030`のlow-precision
prefillを次の順序で改善する。

1. 共通化可能なtile、K staging、half2 dot、scale適用、output mappingをMXFP8で実装・検証し、採否を決める。
2. MXFP8でcorrectness／resourceを満たした共通候補をMXFP6へ接続し、MXFP6内で別に採否を決める。
3. 共通経路の判断後に、packed E3M2（4 value／3 byte）の読込みとFP16 pair展開というMXFP6固有部分を追加し、採否を決める。
4. 各段階のcontrol／candidate、採用／benchmark-only／棄却、resource、profile、full-model結果を同じ表へ整理する。

Phase 74のID47は32x32 output tile、4 output/thread、K32 staging、exact E3M2→FP16、half2 dot2で
gfx1030 MXFP6を改善したが、64x64／128x32／128x64、16〜32 accumulator/thread、K64〜256 staging、
ID47内のpacked E3M2x4 ingressは未評価である。Phase 75はこれらを一度にMXFP6専用kernelへ足さず、
byte-aligned E4M3を持つMXFP8でschedule差を先に分離する。

## 固定scope

- hardware: canonical Radeon Pro V620、exact `gfx1030`、wave32。GPU UUID／BDFはPhase開始時preflightで再固定する。
- software: ROCm 7.14.0、AMD clang 23、Code Object V6、target別release build。
- operation: dense modelのsingle-request prefill matmul。decode `M=1`、attention、GDN、KV、samplingは変更しない。
- MXFP8: OCP E4M3FN W8A8、block 32、E8M0 scale、FP32 accumulation、BF16 RNE output。
- MXFP6: OCP E3M2 W6A6、block 32、E8M0 scale、FP32 accumulation、BF16 RNE output。
- resident weight／activationは各形式のpacked表現を維持する。whole-modelまたはrequest全体のFP16／BF16／FP32展開、
  persistent展開weight、FP32 attention／KV planeは追加しない。
- primary modelは固定Qwen3.5-4B dense GGUFとし、MXFP8は
  `sha256:f253d9f47603d84718b4fdb898b434e493d732b52838ba9abfdfafe73a5d076f`、MXFP6は
  `sha256:d0ff2e1de9d87dddddcde8f85ef305bbf21a06d5f7586d077ba1178580a0264e`、明示FP16 KVを使う。
- model共通性の最終補助行は、既存reviewed Qwen3.5-27B MXFP6 GGUF
  `sha256:d1142468252af487d52ebf72a29a4bb62487a635c174e709bebd73b0c337a82c`の512 inputだけとする。
- `gfx1201`、`gfx942`、gfx1031〜1036、RDNA3、CUDA、CPU、MoE、別model architectureへ性能結果を一般化しない。
  共通sourceに触れた場合のhost selector／target別compile回帰だけを確認し、別GPUのfull-model再測定は行わない。

## Phase開始時baseline

- MXFP8 controlはPhase 69 ID41
  `matmul.mxfp8.w8a8.gfx1030.mmq-col8.vector32.v1`とrow8 rollbackである。履歴上の4B 3+10中央値は
  512／2,048 inputで`254.4461／249.3441 tok/s`だが、採否にはPhase 75の同一source／artifactで取得するfresh controlを使う。
- MXFP6 controlはPhase 74 ID47
  `matmul.mxfp6.w6a6.gfx1030.half2.32x32.v1`と旧ID25 tiled16 rollbackである。最終既定値は
  512／2,048 inputで`411.82／393.60 tok/s`、matrixはfresh profileのkernel duration `92.62%`を占めた。
- fixed llama.cpp Q6_KのPhase 74値`2,077.47／2,061.67 tok/s`はcross-format残差の参考としてだけ再掲できる。
  MXFP8／MXFP6 candidateのstrict controlやPhase 75の必達値にはしない。
- source commit/tree、dirty diff、compiler、runtime library、GPU identity、model lock、input token file、command、binary／code object
  SHA-256を最初に固定し、historical absolute値を新candidateとのpaired比較へ流用しない。

## 共通実装境界

新しいhalf2 software-MMQ familyは、少なくとも次のcompile-time policyへ分ける。

- `FormatIngress`: row stride、value group load、形式固有codeからFP16 bitsへの変換を所有する。
  - MXFP8は4個のE4M3 byteを1回の32-bit loadで読み、4個のFP16 valueへ展開する。
  - MXFP6移植段階はまずID47相当の既存scalar E3M2 ingressを使い、schedule差と6-bit最適化を混ぜない。
  - MXFP6固有段階だけが3 byte／4 code group loadとx4 FP16展開を使う。
- `TilePolicy`: output rows、output columns、threadからoutputへのmapping、accumulator数を所有する。
- `KStagePolicy`: K32 blockを何個stageするかとbuffer配置を所有する。K64以上でも各K32のE8M0 scale pairを
  contributionへ適用してから累積し、異なるscale blockを先に合算しない。
- 共通body: half2 dot2、FP32 block sum／accumulator、E8M0 scale適用、BF16 RNE store、M/N tail処理を所有する。

MXFP8とMXFP6はformat、resident layout、provider／kernel identity、selector、rollbackを別に維持する。
sourceを共有しても同じtileを両形式へ強制しない。各形式・shapeで独立に`scoped default`、`benchmark-only`、`rejected`を決める。

## 候補の評価順

候補をCartesian productで一括生成せず、次の一変数順で絞る。

1. K32固定で32x32 control、64x64、128x32、128x64を比較する。64x64／128x32は16 output/thread、
   128x64は32 output/threadを基本案とし、実thread mappingで値が変わる場合は計画履歴へ明記する。
2. 最良のcorrect geometryだけでK32 controlに対するK64、K128、K256を比較する。
3. 最良のK depthだけでsingle bufferとdouble bufferを比較する。resourceまたはkernel時間が支持しなければdouble bufferは棄却する。
4. MXFP8の結果を閉じてからMXFP6へ共通候補を移植する。
5. MXFP6共通候補の結果を閉じてからpacked E3M2x4 ingressを追加する。

correctness failure、不正なglobal read、launch不能、fallback発生はそのcandidateを`rejected`とし次段へ渡さない。
数値・resourceは正しいがMXFP8既定値を上回らない候補は`benchmark-only`とし、format interactionを切り分けるため
最良の一候補だけをMXFP6へ移植できる。MXFP8のproduction selectorはこの場合変更しない。

## 作業単位

### P75-A: identity、fresh baseline、runner拡張

1. V620をstable UUIDで単独可視化し、foreign workload、ECC、ROCm loader root、target、wave sizeをpreflightする。
2. 同じPhase開始sourceでMXFP8 ID41とMXFP6 ID47のoperator、resource、512／2,048 full-model、512-token profileを取得する。
3. runnerへcandidate kernel ID／symbol、actual dispatch、fallback、cleanup、median／MAD、kernel時間、resourceを同じschemaで出す列を追加する。
4. production projectionとしてM=`128/512/2048`、K/N=`2560/9216`と`9216/2560`、N=`1024`を含める。
   selector境界はM=`127/128/129`、K=`2016/2048/2080`、N=`1023/1024/1025`とし、非整列M/N tailも含める。
   K=`31/32/33`は現行契約どおり非32整列をfail-closeすることを確認する。

### P75-B: MXFP8で共通half2骨格を検証・判定

1. E4M3FN finite codeをFP16で同値に表し、signed zero、subnormal、最大有限、NaN classとE8M0 scale `255`の既存伝播規則を
   維持するdevice-inline ingressを追加する。全256 code、4 lane、activation／weight両operandをhost／device oracleで確認する。
2. ID47の2次元half2構造を形式非依存bodyへ抽出し、MXFP8 32x32／K32を最初のinstantiationとする。
   current ID41をproduction controlとして保持し、共通化だけで既存symbol／selectorを置換しない。
3. 上記の一変数順でoutput geometry、K depth、bufferingを測る。各候補についてwave、workgroup、LDS、SGPR、VGPR、
   scratch／spill、static dot2、dispatch数、kernel medianを記録する。
4. 独立FP32 oracle、BF16 error、nonfinite位置、repeat determinism、tail、HIP-only、fallback false、cleanupをPASSした候補だけを
   4B draft 1 warmup＋3 measuredへ進める。最終候補は同一binaryで3 warmup＋10 measuredのcontrol/candidate/controlを取得する。
5. MXFP8の判断を`scoped default`、`transfer-only benchmark`、`rejected`へ固定してからP75-Cへ進む。

### P75-C: 共通候補をMXFP6へ適用・判定

1. P75-Bで選んだTile／KStage policyをMXFP6へinstantiationし、最初はID47と同じscalar
   `packed_e3m2_at`相当のingressを使う。3-byte group loadやSWARをこの段階へ混ぜない。
2. 現行ID47を別symbolのcontrol／rollbackとして残し、refactor後32x32 control、選定geometry、選定K depthを同一runnerで比較する。
3. 全64 E3M2 code、4 packed lane、signed zero、最大有限、入力Inf saturation、NaN block、E8M0 scale
   `0/1/118/127/134/254/255`、tail、repeatを独立FP32 oracleで確認する。
4. operatorと4B 512／2,048のcontrol/candidate/controlを取得し、MXFP6内で`scoped default`、`benchmark-only`、`rejected`を決める。
   MXFP8で採用したtileがMXFP6で負けても、形式固有selectorを分けたまま結果を棄却として閉じる。

### P75-D: MXFP6固有packed ingressを追加・判定

1. activation／weightの各4 valueについて同じ3 byteをscalarごとに再構成せず、境界外over-readのない
   24-bit group loadを一度だけ行うcandidateを追加する。
2. 4個の6-bit codeを取り出して4個のexact FP16 bits、または2個のhalf2 pairへ展開する。まずgroup loadだけを分離し、
   追加SWAR／packed storeはprofileで変換命令が残差になった場合だけ別candidateとして比較する。
3. P75-Cの選定schedule、scale適用、accumulation tree、output mappingは固定し、共通schedule改善と6-bit ingress改善を混ぜない。
4. 全64 codeを全laneへ混在させたpacked oracle、row終端、K32 block境界、production shape、resource、4B 512／2,048を比較し、
   `scoped default`、`benchmark-only`、`rejected`を決める。算術treeが同じcandidateはcontrolとのBF16 digest一致を要求する。

### P75-E: 最終benchmarkと結果整理

1. 同一最終sourceから、少なくとも次の段階を一表へ並べる。
   - MXFP8 ID41 fresh control。
   - MXFP8 common-half2 32x32 controlと、P75-Bの最終候補。
   - MXFP6 ID47 fresh control。
   - P75-Cのshared schedule候補。
   - P75-DのMXFP6 packed-ingress候補。
2. primary 4Bは512／2,048 input、最大4 output、greedy、ignore EOS、明示FP16 KV、3 warmup＋10 measuredとし、
   prefill tok/s、prefill秒、E2E、median／MAD、kernel share、resident／peak VRAM、dispatch、生成tokenを記録する。
3. 最終MXFP6既定候補または維持されたID47で27B 512 inputを1 warmup＋3 measuredし、4B専用selectorでないことを補助確認する。
4. Phase 74のfixed llama.cpp Q6_K値を参考列として併記する場合はformat差を明記し、Phase 75候補のstrict A/Bへ混ぜない。
5. 採用source、維持したrollback、benchmark-only／削除したcandidate、selector scope、code object／binary identity、
   correctness、resource、残差をmatching historyと追跡summaryへ同期し、計画をarchiveへ移す。

## 数値分類

- MXFP8 E4M3FN finite valueからFP16への展開自体は実数値を維持する。half2 dot、output tile、K stagingにより
  accumulation treeがID41から変わるcandidateはN1として扱う。
- MXFP6を共通Tile／KStageへ移すcandidateも、ID47とtreeが変わる場合はN1とする。各K32 scale pairを維持し、
  独立FP32 oracle、最大absolute／relative error、nonfinite位置、repeat、生成tokenを記録する。
- P75-Dのpacked E3M2x4 ingressは、同じscheduleでscalar ingressと全値・全laneが一致するN0候補とする。
- 旧KV default用のtop-1 `0.99`閾値をW/A providerへ流用しない。N1の差は観測して記録するが、固定速度向上率や
  cross-format top-1を新しい自動品質gateにしない。

## 採否と完了条件

- P75-B、P75-C、P75-Dの各段階で、少なくとも一つの実kernelをexact gfx1030で数値検証・性能測定し、判断を記録する。
- production採用は同一binaryのpaired control/candidate/controlで候補が両controlより安定して速く、primary 4Bの
  512／2,048両行を退行させず、provider identity、fallback false、cleanupを確認できたshapeだけに限定する。
- 固定の改善率は置かない。候補が勝たない場合はMXFP8 ID41／MXFP6 ID47を維持し、試行結果と残差が再現可能ならPhaseを完了できる。
- selector keyはexact target、format／layout、M/N/K、alignment、resource条件だけとし、model名、layer番号、prompt、token、
  benchmark結果を入れない。unsupported shapeは既存providerへ明示的に戻し、実行失敗後のsilent fallbackは追加しない。
- 最終結果表で「MXFP8共通部分」「MXFP6への移植効果」「MXFP6固有ingress効果」「全体prefill効果」を分離できることを完了条件とする。

## 対象外

- `gfx1201` WMMA経路の再最適化、ID48変更、NVFP4／MXFP4／BF16／FP8 weight、KV形式、attention、GDN、decode。
- persistent FP16／BF16／FP32 weight、request全体の展開activation cache、FP32 attention／KV保存。
- llama.cpp Q6_Kの製品対応、integer `dp4a`式の移植、外部engine codeのcopy。
- model architecture追加、MoE、batching、multi-GPU、Infinity Fabric、RDMA、WebUI/API変更。
- MXFP8／MXFP6 quantization recipe、GGUF encoding、public ABI、default KV、quality policyの変更。

## 完了結果

2026-09-03にP75-A〜Eを完了した。canonical V620 exact `gfx1030`で、MXFP8は128x64／K32／double-buffer
half2 ID55、MXFP6は同じscheduleとpacked E3M2x4 ingressを持つID57を既存のdimension-only scopeへ限定採用した。
MXFP6 scalar-ingress ID56は共通scheduleの効果を分離したbenchmark candidateとして保持し、MXFP8 ID41／MXFP6
ID47は明示rollbackとして維持した。

同一最終binaryのQwen3.5-4B、3+10 control/candidate/controlで、MXFP8 ID55は512／2,048入力を
`993.6765 / 1,104.1643 tok/s`、MXFP6 ID57は`1,008.7235 / 1,095.3894 tok/s`とし、各候補は両controlより
安定して速かった。27B MXFP6 512入力も強制指定なしで`157.7535 tok/s`をPASSした。全行で生成token、VRAM、
dispatch、HIP-only、fallbackなし、cleanupを維持した。

全256 E4M3FN code／全64 E3M2 code、lane、特殊値、tail、独立FP32 oracle、10-repeat operatorをPASSした。
ID55／56はaccumulation treeを変更するN1、ID57はID56と全operator digestが一致するN0である。最終resourceは
ID55がLDS 26,112 B／SGPR 38／VGPR 156、ID57が26,112 B／40／151で、private／spillは0だった。
gfx1201 wave32とgfx942 wave64はtarget別compile-onlyをPASSし、別GPUのperformance claimは追加していない。

詳細な候補表、最終benchmark、profile、artifact identity、制約はmatching historyと追跡要約を正本とする。

[全体計画](../../../../main-plan.md) /
[Phase 37以降のロードマップ](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md) /
[Phase 74保存済み計画](../../../../archive/2026/09/1-10/phase74-mxfp6-prefill-llama-optimization-loop.md) /
[履歴](../../../../../history/2026/09/1-10/phase75-gfx1030-mxfp8-first-shared-half2-optimization.md) /
[追跡要約](../../../../../../ci/matrix/phase75-gfx1030-mxfp8-mxfp6-shared-half2-v1.json)
