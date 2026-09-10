# Phase83.5: MXFP8・固定sampling・MTP最適化の履歴

> 状態: 実装・検証完了。速度目標は2026-09-10ユーザー指示により緩和した。公開commitのCI結果はGitHub Checksで確認する。

試行時の「未完了」「次に実行」は当時の記録であり、最新状態は[main-plan](../../../../plans/main-plan.md)と[実行計画](../../../../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)を参照する。


## 最終検証と制限（2026-09-10）

- 速度目標はユーザー指示で参考値へ緩和した。R56までの追加最適化を保持し、速度追求を終了する。初回通常測定のR9700 decode24.435 tok/sと旧25目標との差を隠さない。
- CI修正はRustの引数数lint属性、不要な借用・collectの整理、Rust/C++整形とmanifest同期に限定した。C++整形には独立includeの並べ替えと同じ文字列の分割がある。演算・sampling・state契約を変更せず、変更前後の21source hashを照合し、両targetをfresh buildした。
- R56 APIは両GPUで5case（短文、SSE、MTP進行後cancel、recovery、8192/128）をPASS。cancel時に提案2・採用2を確認し、長文textは各GPUのR56 benchmarkと一致した。HIP-only、fallbackなし、shutdown cleanup0。新ビルドとの対応は上記lint/format差分のreviewに基づき、最終benchmarkは新ビルドで別途実行する。
- API最大VRAMはV620 32,135,843,840 B、R9700 32,253,530,112 B。各32GB級GPUの物理VRAM以内。R9700の元serviceはhash/unit/UUIDを維持し、healthz/readyz200、client0へ復元した。
- H1は1,539 selected、H2は38 selectedをPASS。H0初回は627 selectedのうち623 PASS、4失敗はRust/C++整形、Clippy、RMSNorm H3 tupleの更新漏れ。canonical全features Clippy、整形とmatrixを修正後に確認し、失敗を残したまま成功扱いにはしない。
- BF16 full-model品質同等性は未証明。55.56GB/18shardのBF16 full-model入口・lock整備やTP/offloadをこのPhaseへ追加しない。公開runtimeの数値・state・sampling oracleと出力比較を、それぞれの検証範囲に限定して報告する。
- 次のPhase84はMTP重み量子化。従来の他精度最適化はPhase85、batchingはPhase86へ移した。companion全configurationの厳密な総reservation、tools完全対応、vision、TP/batching、全モデルへの適用は今回の完了範囲に含めない。

- 最終buildの製品CLI `sllm generate`も両GPUで26入力/17出力をPASS。固定T1/P.95/K20、MXFP8 KV、MTP提案12/採用10/棄却2、HIP-onlyを確認した。R9700 service復元を確認し、V620は最終CLI後VRAM17,215,488/GTT15,020,032 B、GPU idleへ戻った。
- V620比較器の初回判定は、MTP有無の出力一致を誤って要求してFAILした。測定は再実行せず、承認済み条件に合わせMTPあり対R56と各modeの反復一貫性へ訂正した。rootもraw結果から独立にtoken/採否/中央値/MADを確認した。初回失敗と訂正理由は最終比較記録に残す。
- 累積reviewでcorrectness/security blockerなし。既存のadaptationは従来noticeで追跡し、このPhaseの新規直接importはない。公開手順ではこのcandidateをcommit/pushし、同commitのhost・basic/public-runtime H3を確認する。

## 公開CIでの修正（2026-09-10）

- 初回公開commit `3dbee53ba455f15051411efd68c601b31f27ad17`のhost H0/H1/H2とbasic H3は成功した。
- [public-runtime H3](https://github.com/jyohukuchan/sLLM/actions/runs/34452726526)は両targetでコンパイル・リンク後に失敗した。GQA6 W16 kernelのbool template付きHIP起動関数2種類が、正当なdevice stubの有限リストに未登録だった。
- 実ELFと照合した2symbolだけを追加し、未知のstubを拒否する回帰テストを追加した。CPU代替や未知のstubを一括許可していない。source manifestを同期し、修正commitのGitHub Checksでhost・basic/public-runtime H3を確認する。
- 2回目の[public-runtime H3](https://github.com/jyohukuchan/sLLM/actions/runs/34453953479)ではstub検査を通過し、続く依存関数検査で`hipblasLtGetGitRevision`・`hipblasLtGetVersion`の登録漏れを検出した。runtimeが実際に使う2関数を厳密な依存リストへ同期し、欠落・余分・重複・host定義の拒否を維持する。既存native objectのHIP依存も照合し、H3対象外のevidence runtime由来3関数は追加しない。残るhost ELF検査条件を確認し、最終判定は修正commitのCIで行う。
- この修正はCI検査と記録だけで、GPU演算・runtime・build入力を変更しない。最終candidateのGPU証拠は公開commitとのsource hash対応を確認して再利用する。

## 最終比較（同一candidate、8192入力/128出力）

各行1 warmup＋3 measured。数値はmeasured中央値で、prefillはcompanion準備、decodeはdraft/verify/reject/replay/sampling/finishを含む。各回新しいtarget/companion requestを作成し、prefix cacheを再利用しない。

| GPU | MTP | prefill tok/s | decode tok/s | TTFT ms | E2E ms |
| --- | --- | ---: | ---: | ---: | ---: |
| V620 gfx1030 | なし | 224.891 | 14.325 | 36446.276 | 45305.110 |
| V620 gfx1030 | あり | 216.571 | 25.409 | 37855.858 | 42865.124 |
| R9700 gfx1201 | なし | 550.149 | 19.083 | 14908.503 | 21559.177 |
| R9700 gfx1201 | あり | 541.402 | 34.541 | 15163.792 | 18870.292 |

- MTPによりdecodeはV約1.774倍、R約1.810倍。E2EはV約5.4%、R約12.5%短縮した。prefillはMTPありの方が遅く、decode倍率をprefillや全体時間の倍率と呼ばない。
- MTPあり採用率はV74/106（69.8%）、R78/100（78.0%）。全反復でR56のtoken/text hashと採否・dispatchが一致した。
- MTP有無の同じ位置のtoken一致はV7/128、R2/128で、両方2token目から分岐する。固定seedでのsampling経路差を含む比較であり、品質劣化の測定ではない。
- R9700の初回MTPありdecodeは5172.320 ms（約24.55 tok/s）、measuredは3660.574/3676.746/3706.439 ms。初回と反復の差を新しい最適化の効果へ帰属しない。過去の単回未達は当時の観測として保持し、反復条件での未達証明にしない。
- この最終反復値は旧参考目標200/20・500/25を上回る。ただしユーザー承認による速度条件緩和と追加追求終了の判断はそのまま記録する。
- [最終candidate・source対応・raw結果digest・MAD・API記録](phase83-5-closeout-evidence.json)と[Phase82/83初回比較](phase83-first-request-comparison.json)を併記する。過去の単回値と今回の反復中央値を同じ測定条件と呼ばない。

## 試行と訂正の時系列

### 着手時の探索（2026-09-09）

- V620-A、Phase83 r26 binary、ID87／ID88／QTILE4／ID93を明示有効、MTP幅2、
  同じ8192／128 fixture、warmup 0／measured 1でprefill **194.161**、decode **11.581 tok/s**。
  TTFT **42.231秒**、E2E **53.197秒**、draft採用 **72/112**、proposal block **56**。
  target／draftともHIPのみ、fallbackなし、terminal非有限logit 0、cleanup zero。
  raw結果は`.local-artifacts/phase83-5/v620a-r26-existing-candidates-r1/result.json`。
  r26はPhase83で記録済みの整形／lint変更との対応を持つbinaryであり、現在HEADの再buildとは呼ばない。
  opt-in構成の単回探索なので既定採用・正式速度達成・BF16品質同等性は未認定。
- R9700もr26、ID89／ID90／QTILE4／ID93、MTP幅2、同じ単回条件でprefill **354.226**、
  decode **13.812 tok/s**、TTFT **23.162秒**、E2E **32.357秒**、draft採用 **73/108**。
  target／draftともHIPのみ、fallbackなし、terminal非有限logit 0、cleanup zero。
  raw結果は`.local-artifacts/phase83-5/r9700-r26-existing-candidates-r1/result.json`。両targetとも目標未達。
- V620の同構成profileではdecode GPU区間約10.480秒／区間全体11.871秒。
  残るFP8 half2小M kernelが合計約3.539秒、NVFP4 rowgrid約2.050秒、staged32 attention stage1約1.935秒。
  prefillはID87約14.998秒、QTILE4約11.162秒、FP8 half2 64x64約9.700秒。
  profileの区間はMTP prefix primingを含め、計測による速度への影響があるため正式性能へ使わない。
  raw trace／分析は`.local-artifacts/phase83-5/v620a-r26-existing-candidates-profile-r1/`。
- llama.cpp比較から、通常draftが使わない全語彙logits readbackを省く修正を第一候補とした。
  token選択とhidden取得を維持し、診断用logits取得APIは残す。
  partial accept時のtarget prefix再実行削減も候補だが、GDN stateを含むaccepted位置の復元が必要であり、
  単なるKV suffix削除で代用しない。新しいprofileで残りの費用を確認して優先順位を決める。
- 第一修正は通常MTPのfull logits readback除去と、exact geometryのQTILE4既定化。
  両targetのrelease buildは成功し、各buildの460入力fileは開始／終了間で不変だった。
  V620 r1ではQTILE4 flagを未設定にして同じ残りの候補を使い、生成128token・MTP採否はr26候補と一致、
  HIP-only、非有限logit 0、cleanup zeroを維持した。prefill **188.949**、decode **11.623 tok/s**。
  単回差を有意な速度改善とは認定しない。raw比較は`.local-artifacts/phase83-5/r1-comparison.json`。
  旧benchmarkの固定audit一覧にQTILE4が欠けていたため、次buildでは同providerの実dispatch数と
  ID87／89／88／90／93等のselector環境値をreportへ追加する。
- QTILE8／8-wave候補はV620-Bのquery127／128／129、prefix0／31／256で独立oracle最大1 ULP、
  QTILE4とbitwise一致だったが、prefix256では約1.53〜1.54倍遅く、長いprefillへの採用を棄却した。
  gfx1030のVGPRは106→162、spill 0、LDS 2048 byteで、register増加によるoccupancy低下が原因候補。
  `.local-artifacts/phase83-5/qtile8-experiment/`に実験source・数値・時間・compile metadataを保持する。
  続く16-wave版はquery/thread比を維持し、V620-Bの8ケースで独立oracle最大1 ULP、QTILE4とbitwise一致。
  prefix7168／query1024では153.762→139.853 msだった。gfx1030、MXFP8 E4、q24／kv4／d256、
  query128以上・prefix1024以上へ限定した既定候補を実装した。明示QTILE4=1は従来Q4、=0はgenericへ戻す。
  scratchの両target compileとV620数値検証は済んだが、Phase7 compile probeは新kernelを含まない。
  productionのcompile／公開runtime数値／モデル性能はr2 build以降で確認する。
- 第二候補ではID87／89・ID88／90・ID93の対象shapeを限定した既定化と、ID92の
  K6144／N5120・M2〜4対応を実装した。追加FP8経路は既存fused bodyを使い、公開runtime経由の
  M1〜5・全出力BF16 oracle回帰テストを追加した。両targetのr2 release buildを開始した。
- r2 release buildは両targetとも成功し、461入力fileの開始／終了hashが一致した。
  V620の公開FP8テストはM1〜5の全出力がoracleと一致、M2〜4のID92とkernel symbol、cleanupを確認した。
  `.local-artifacts/phase83-5/fp8-smallm-public-r2/`にarchive／test hashと実行結果を保持する。
  新kernelの実objectからCI symbol inventoryを更新し、H3契約31 test／463 subtest、JSON manifest検証を通過した。
  追加Rustテストのceil式だけをlint修正し、clippyとKV state 29 testも通過した。
- V620 r2の最適化flag未設定・MTP幅2・warmup0／measured1はprefill **198.317**、decode **12.403 tok/s**、
  TTFT **41.347秒**、E2E **51.587秒**、draft採用 **73/110**。Q4 16回／Q8W16 112回と
  ID87／88／92／93の実dispatchを確認し、HIP-only、terminal非有限logit0、cleanup zeroだった。
  raw結果は`.local-artifacts/phase83-5/v620a-r2-defaults-width2-r1/`。目標200／20は未達で、
  単回探索を正式反復の代用にしない。MTP幅によるdraft／verify／replay費用の違いも次に測定する。
- 同じV620 r2でMTP幅1はprefill **195.316**／decode **13.470 tok/s**、採用57/70、
  幅3は **195.531**／**12.626 tok/s**、採用83/135だった。両方ともPASS、128確定出力。
  この単一fixtureでは幅1が最速だが、幅選択だけでは20 tok/sへ届かない。最終既定幅は未変更。
- R9700 r2の同じflag未設定・幅2・単回条件はprefill **353.451**／decode **13.878 tok/s**。
  ID89／90／93とFP8 native ID5の実dispatch、HIP-only、cleanup zeroを確認した。
  raw結果は`.local-artifacts/phase83-5/r9700-r2-defaults-width2-r1/`。500／25の目標は引き続き未達。
- R9700 ID89のstage K64候補は初回scratchでM65以上が非有限・非決定的となり不採用。
  scale stagingが256threadの単回storeのままで、必要な512 activation scaleの後半を未初期化にしていた。
  strided loadへ修正したscratchを再検証する。修正後は127 VGPR／spill0であり、修正前113 VGPRの
  resource値を採用候補の値として再利用しない。productionにはまだ反映していない。
- K64再検証ではscratch harnessの余分なゼロ寸法caseも除去した。修正後のM65境界と
  両tupleのM17／128／1024はfinite・repeat・sample oracle最大0 ULP、controlと全出力hash一致。
  M1024では約10〜15%速いが、wide M17／65／128では約2〜3%遅いため一律採用はしない。
  失敗と修正後の証拠は`.local-artifacts/phase83-5/nvfp4-prefill-experiment/corrected-r2`／`corrected-r3`へ分離した。
- MTP all-accept後と外部proposal後の状態整合に、既存state-only APIを接続して不要なLM head／argmaxを除いた。
  r3は両target build、関連host test、frontend clippyを通過した。V620-Bモデルrunはr2 V620-Aと
  128token・採否が一致、draft dispatchは20380→20287。異なるGPU個体のため速度差は変更効果へ帰属しない。
  R9700 r3 profileもr2と128token一致、draft dispatchは20299→20209。profileのinclusive kernel時間は
  ID89 prefill約12.167秒、Q4 prefill約6.508秒、NVFP4 small-M decode約2.811秒、主要FP8 native decode約1.988秒。
  intervalの重なりがあり、これらの合計やtraceの隙間を直列の削減可能時間／CPU idleと解釈しない。
  R9700 production serviceは元のbinaryで復帰し、healthz／readyz 200を確認した。
- NVFP4の4-row共有LDS候補はV620-Bの両tuple・M2〜4で全出力oracle／control／repeat一致だったが、
  全ケースで遅く、control/candidate速度比0.191〜0.522で棄却した。新しいstage同期が原因候補。
  `.local-artifacts/phase83-5/nvfp4-smallm-fused/`に失敗案を保持し、同期を増やさないregister共有案へ絞って調べる。
- 残るFP8のQ/K/V投影と最終8層MLPから4 tupleを特定し、ID92 fused scratchを追加検証した。
  4 tuple×M2〜4の12ケースで全出力ID71一致、独立oracle 144点で最大0 ULP、単体で約4.4〜10.6倍。
  この改善率をモデル全体へ適用せず、exact tupleに限定したr4の公開経路統合・数値・性能を進める。
- llama.cppのrecurrent state plane／bounded rollbackを追加調査した。sLLMのGDNは現状2-slotで
  transition終端だけを保持するため、partial acceptの再実行をKV trimだけで除けない。
  M3の中間2位置を保持する場合、48層のconv＋recurrent stateで追加293.625 MiBとなる。
  pinned llama.cppのspeculative初期化自体は`n_rs_seq=0`であり、snapshotをMTP既定で使う実装とは扱わない。
  checkpoint導入は候補であり、まだ実装・性能改善の証拠はない。


- r4のFP8公開GPU回帰はV620-Aで5追加tuple×M1〜5×3反復、75実行が全出力oracle最大0 ULP、
  selector／device symbol／cleanupを通過した。証拠は`.local-artifacts/phase83-5/fp8-smallm-public-r4/`。
  V620-Aの8192入力／128出力・幅2・chunk1024・warmup0／measured1はprefill **198.308**、
  decode **16.766 tok/s**、draft採用74/108。HIP-only・非有限0・cleanup zeroを確認した。
  同じr4の幅3は **196.263／15.828**、幅2・chunk2048は **197.158／16.717 tok/s**で、
  この探索では幅増加／chunk拡大による改善は得られなかった。正式1 warmup＋3 measuredは未実施。
- NVFP4 small-Mのregister row再利用候補はV620-Bで200ms程度の連続warmup後、20組のAB／BA交互比較を行った。
  両tuple・M2〜4の全出力oracle／control／repeat最大0 ULP、wideは1.155／1.339／1.443倍、
  downは1.104／1.240／1.350倍。register増加だけを理由とした初期棄却判断は撤回し、
  exact gfx1030・M2〜4・2 tuple限定のID94として統合中。明示ROWGRID=1はID88を保持する。
  証拠は`.local-artifacts/phase83-5/nvfp4-smallm-vgpr-paired/`。全モデルでの改善はまだ未検証。
- R9700 ID89のN32派生はwideで遅く、downの改善も一貫しないため棄却した。
  K64／N64はreverse-order・200 warmup比較でM256／512／1024がwide 6.8〜10.8%、
  down 4.3〜14.4%短縮した。M>=256の2 tuple限定で統合する。
  K32／K64を同じdevice wrapperに入れる実装はLDSが合算され23,040 bytes／159 VGPRとなるため撤回し、
  logical ID89を維持した別device symbolへ分割した。実production compileでK32は7,680 bytes／118 VGPR、
  K64は15,360 bytes／127 VGPR、両方spill0。M255／256／257のGPU境界検証は未実施。
- V620のMXFP8 attentionで8 keyずつLDSへ置くscratchは6ケースでQ4とbitwise一致、
  独立oracle最大1 ULP、Q4比の時間比0.934〜0.967だった。ただしr4既定はQ8/W16であるため、
  採用判断にはQ8/W16との直接比較を行う。Q4比だけを既定経路に対する改善と扱わない。
- R9700 serviceは追加のQ8/W16／K64検証のため再度停止中。検証後に元のproduction binaryで
  復旧し、healthz／readyzを確認する。以前の復旧記録を現在の稼働証拠として扱わない。

- K64のscratch境界検証はR9700のM255／256／257・両tupleで完了し、finite、sample oracle最大0 ULP、repeatとK32との全出力hash一致を確認した。旧symbol executableはStageK32固定の比較対照であり、新host dispatcherの証拠には使わない。r5公開runtimeでのdevice symbol確認を続ける。
- ID94を公開providerへ変換する`matmul_runtime.inc`のcaseが統合時に欠けていたため追加した。kernel selectorだけで完了とせず、r5公開plan／executeから検証する。

- MXFP8 Q8/W16 stage8は現在のstage1既定との直接AB／BA比較で6ケースすべて2.14〜2.72%遅かった。全出力bitwise／repeat／oracleはPASSしたがgfx1030採用は棄却し、productionはstage1を維持する。Q4比だけなら改善に見えたため、比較対象を現在の既定へ揃える必要がある。証拠は`.local-artifacts/phase83-5/attention-key-stage8/stage1-vs-stage8-summary.md`。

- 追加probe終了後、R9700 serviceを元のproduction binary／設定で復旧し、healthz／readyz 200を再確認した。service-stateのrestore_requiredはfalse。

- r5統合時、host selector testに残っていたFP8 ID71／NVFP4 ID88期待を新ID92／94へ揃え、production public runtime host fault testとRust Q8 metadata testがPASSした。Q8/W16のgfx1201既定拡張もsource freeze済み。
- CIは新includeをCargo rerun登録・direct compile inventory・matrix/artifact schemaへ反映し、device symbolsを162個へ更新した。途中のschema更新漏れ／symbol順序不一致を修正後、H3契約31 test＋466 subtestがPASS。r5両target release buildは進行中。build開始後の変更はhost test期待値だけで、完了時にsource mappingで確認する。

- r5両target release buildはexit0。開始／終了のsource差はhost test期待値だけでproduction／build入力の変更なしをidentityへ記録した。V620-B公開ID94テストは6ケース、全出力独立BF16 oracle最大0 ULP、repeat、ID94／device symbol／grid、fallbackなし、cleanupをPASS。ID88別実行との比較はscratch証拠であり、この公開test自体には含めない。
- r5のflag未設定・MTP幅2・chunk1024・warmup0／measured1はV620-Aがprefill **197.570**／decode **17.806 tok/s**、TTFT41.503秒／E2E48.636秒、R9700が **383.547／13.794 tok/s**、TTFT21.393秒／E2E30.600秒。両方HIP-only、非有限0、cleanup zero。V620はr4、R9700はr2と128出力token hashが一致し、採用は各74/108・73/108。Q8/W16実dispatchは両方112回。目標未達のためV620幅1とR9700 small-M候補を調べる。正式反復はまだ行っていない。

- r5 V620-AのMTP幅1・単回はprefill196.915／decode16.785 tok/s、採用57/71、HIP-only／cleanup PASS。幅2より遅いため既定幅は2を維持する。chunk512・幅2の追加探索を開始した。

- r5 V620-A chunk512・幅2はprefill **188.427**／decode **18.127 tok/s**。prefillが悪化するため採用せず、chunk1024の比較条件を維持する。これは単回探索でありdecodeの小幅差を改善と認定しない。
- r5 R9700公開K64は両tuple・M255／256／257で新旧device symbolの切替、sampled 5×5境界oracle最大0 ULP、全出力finite／repeat／cleanupをPASS。公開MXFP8 attentionもQ8/W16切替境界を含め最大1 ULP、Q4比較0差をPASSした。K64 testはscratchからbyte一致でtracked testへ保存し、任意CMake targetを登録・configure確認した。r5 kernel binaryの変更はない。

- Phase82比較用の両target binary／同一8192 token fixture／1 warmup＋3 measured結果が保存済みであることを再確認した。`.local-artifacts/phase83/baseline-v620-mxfp8-8192-r1/`と`baseline-r9700-mxfp8-8192-r1/`はMTPなしのbaselineであり、候補のMTPあり結果と混同しない。候補側の正式反復は未完了。
- R9700のID90→VGPR row再利用scratchは両tuple・M2/3/4の全出力独立oracle／control／repeatが最大0 ULP、finite PASS。200ms連続warmup後20組AB／BAではwide 1.671／2.365／2.816倍、down 1.692／2.327／2.454倍。ID94を同じexact scopeのgfx1201にも適用する統合へ進む。明示ID90 controlを保持し、単体倍率をモデル全体の改善率には使わない。証拠は`.local-artifacts/phase83-5/nvfp4-smallm-r9700-vgpr/`。

- llama.cppのRDNA tile配置を参考にしたV620 ID87 TileM128 scratchは、10ケースで全出力control／repeat一致、境界sample oracle最大0 ULP、finite PASS。M512／1024のwideは1.028／1.047倍、downは1.060／1.063倍だったが、M129では両tupleとも約21%遅かった。exact gfx1030・2tuple・M>=512かつM%128==0に限定し、新device symbol／旧logical ID87として統合準備する。小Mと端数は旧TileM64を維持する。152 VGPR／spill0という資源増加だけで棄却せず、実測で範囲を決める。
- V620 FP8 M1の5追加tupleはscratch全出力control一致／finite／repeat／sample oracle最大0 ULP、初回比率1.5〜2.6倍だった。ただしmin/maxに約4倍のclock過渡があるため、200ms連続warmup＋20組AB／BAで再比較して採否を決める。170 VGPRという値自体を不採用理由にはしない。

- r5 V620-A MTP幅3はprefill **198.643**／decode **17.481 tok/s**、採用86/126、HIP-only／cleanup PASS。幅2を上回らないため既定幅2を維持する。
- R9700 FP8 small-M dot2候補は8ケースの全出力oracle／ID5比較／repeatが0差だったが、M1で大幅に遅く、M2/3の利益も小さく変動したため現段階で採用しない。続くNVFP4 M1 register再利用も両tupleで全出力oracle／control／repeat一致だったが、200ms warmup＋20組比較で約2.5%／0.2%遅く不採用。証拠は`fp8-smallm-gfx1201/`と`nvfp4-m1-gfx1201/`に保持する。

- V620 FP8 M1追加5tupleの200ms連続warmup＋20組AB／BAは、wide12288／1024／17408、down6144／17408の順に1.679／2.501／1.514／1.597／1.693倍。測定したfused ID82 bodyのRows=1を独立symbolとして統合し、各170 VGPR／544 bytes LDS／spill0を確認した。旧LDS LUT bodyを呼ぶ最初の統合案は測定対象と異なるため修正した。H1 1510 testとgfx1030 compile-onlyはPASS、公開GPU／full-modelはr6で確認する。Rust側のmetadata検証に旧期待が残るため、r6開始前に整合させる。
- ID87 TileM128とgfx1201 ID94のproduction sourceは統合済み。公開prefill testは`phase83_5_nvfp4_prefill_tiling_public_gpu_test.cpp`へ拡張・改名し、gfx1030のM511／512／513／1023／1024／1025とgfx1201のM255／256／257を対象にした。両targetの構文検査はPASS、追加したgfx1030範囲のGPU証拠は未取得。

- r6両target release buildはexit0。build中の`matmul.rs`差はテスト部分だけで、ID82の5tupleを受け付けるproduction validatorは開始前に反映済み。focused Rust matmul 7 test、format、計画リンク検査はPASS。native公開host testの新規debug buildではstaged／staged32のfake stub不足が見つかったため修正中で、GPU kernel failureとは区別する。
- r6 fresh archiveから作った公開GPU testは、gfx1030 FP8 5tuple×M1..5、gfx1030 NVFP4 prefill両tuple×M511／512／513／1023／1024／1025、gfx1201 ID94両tuple×M2/3/4でPASS。metadata切替、repeat、finite、資源解放を確認し、各testのoracle最大0 ULP。prefill oracleは境界5×5 sampleで、全出力独立oracleとは呼ばない。証拠は`r6-public-fp8-gfx1030/`、`r6-public-tiling-gfx1030/`、`r6-public-id94-gfx1201/`。両GPUのr6 full-model単回測定を開始した。

- r6 flag未設定・MTP幅2・chunk1024・warmup0／measured1はV620-A **203.534／18.164 tok/s**、TTFT40.290秒／E2E47.282秒、R9700 **385.391／16.789 tok/s**、TTFT21.294秒／E2E28.858秒。両方128出力hashはr5と一致、採用74/108・73/108、cleanup zero。V620 prefillは単回で200を上回ったが正式中央値は未取得、decodeとR9700両指標は未達。追加採用後のボトルネック確認のため、両GPUでr6 rocprof kernel traceを開始した。profile中の時間は通常性能と混同しない。比較は`.local-artifacts/phase83-5/r6-comparison.json`。

- 公開host runtimeのstaged／staged32 fake stub不足と、ID94既定化に追従していなかったNVFP4 selector期待3箇所を修正した。新規debug buildと対象ctest 1/1がPASS。fakeはABI・失敗処理のhost契約用で、GPU数値の証拠には使わない。
- r6 rocprofは両GPUで完走した。V620 decodeの大きいkernelはMXFP8 staged32 stage1約1.856秒、NVFP4 ID94約1.202秒。R9700 prefillはID89 Stage64約10.701秒、Q8/W16約6.132秒、decodeはFP8 hipBLASLt主kernel約2.162秒、ID94約1.220秒。各値はprofile内のinclusive dispatch durationであり、重なりを無視した直列合計・通常性能の内訳率とは扱わない。
- R9700 StageK64 lookahead scratchは2 parity LDSとnext-tile raw VGPR先読みを実装し、4ケースで全出力control／repeat一致、finite、64点独立oracle最大0 ULPをPASS。200ms warmup＋20 AB／BAでM256 wide約1.21倍、M257 down約1.16倍、M1024 wide約1.18倍／down約1.26倍。VGPR127→228、LDS15360→30720 bytes、spill0だが実測が改善したため、現Stage64と同scopeで統合する。pinned llama.cppと参照worktreeのHEADは異なったが、参照したMMQ3ファイルは両commitでbyte一致。証拠は`nvfp4-prefill-stage64-lookahead/`。

- r6 V620-A MTP幅1はprefill202.137／decode16.951 tok/s、採用57/71、HIP-only／cleanup PASS。FP8 M1追加後も幅2の18.164を下回るため既定幅2を維持する。
- ID89 lookahead production bodyはscratch測定bodyとコメント・空白を除いて一致することをrootで確認した。gfx1030 compile-onlyとgfx1201 compile/resource、host selectorがPASSしsource freeze。新device symbolをCI169個目として登録、公開tiling testのM256以上期待も更新してr7 gfx1201 release buildを開始した。途中CI hash failureは編集中incの差が原因であり、freeze後にmanifestを再同期して再検証する。

- r7 gfx1201 release buildはexit0、全source before/after一致。freeze後のH3契約31 test＋466 subtestもPASS。公開tiling GPU検査は両tuple・M255／256／257でStage32／lookahead切替、境界5×5 oracle最大0 ULP、全出力finite／repeat／cleanupをPASS。単回モデルはprefill **410.482／decode16.787 tok/s**、TTFT19.992秒／E2E27.557秒、採用73/108、HIP-only／非有限0／cleanup zero、128出力hashはr6と一致した。正式反復・目標達成は未完了。
- R9700 FP8 hipBLASLtのM2/3/4・K5120N6144/N10240は、各10 zero-workspace候補、計60候補で全出力rank0比較／独立FP32 FMA→BF16 oracle／finite／repeatをPASS。既定rank0約90〜132µsに対し高速候補は約33〜46µs、200ms warmup＋20 AB／BAで約2.7〜2.85倍。rank番号をそのまま新policyへ保存せず、現在library version/revisionと測定済みalgorithm identityを照合する方法で採用準備する。未知libraryや候補欠落は既存rank0を維持し、M1／M17の既存policyは変更しない。証拠は`fp8-smallm-gfx1201-rank/`。

- metadata-only GPU queryでhipBLASLt version `100401`（1.4.1）、revision `cd957402`を取得した。N6144はindex123373、N10240はindex123374をM2..4共通で選ぶ方針とし、N10240のindex123372との約0.2µs以下の差を理由にM別tableを増やさない。version/revision/index照合とrank0維持を既存prepare/cacheへ統合中。
- MXFP8 E4 valueとE8M0 scaleを別々にdecodeして乗算するloaderに対し、既存`decode_scaled`へ統合するscratchを準備した。256×256組のhost exhaustiveは有限値bit exact、NaN class一致だがGPUは未検証。gfx1201 staged32はVGPR48→45、code size約82%増、gfx1030は53→48、code size約25%増。register減少だけで採用せず、same-process/same-streamの200ms warmup＋20 AB／BA、独立oracle／fullcontrol／repeatで比較する。gfx1201へGPU leaseを付与し、gfx1030 probeも別scratchで準備中。R9700 production serviceはこの検証batch中停止し、終了後に既存binary／設定で復旧する。

- gfx1201のMXFP8 `decode_scaled` loader候補はsame-stream 4ケースでfull bitwise比較／repeatが0差、finite PASS、独立oracle最大1 ULPだった。一方、Q8/W16は境界1.879→1.923ms／長prefix78.388→80.222ms、staged32 M1は0.198→0.436ms、M3は0.420→0.866ms（AB中央値、BAも同傾向）。増えた整数処理の費用がnative FP8変換の削減効果を上回ったと推定し、gfx1201には採用しない。sourceは未変更、証拠は`mxfp8-e4-decode-candidate/same-stream-gpu-r1.log`。native FP8命令を持たないgfx1030は別途same-process probeで確認する。


- gfx1030 staged32の`decode_scaled` loaderはsame-process AB／BAで8192／8193 prefixのM1が約18%、M3が約30%短縮し、全出力control／repeat一致、独立oracle最大1 ULPを確認した。exact gfx1030・E4・32-split stage1だけへ統合し、gfx1201／8-split／prefillは従来loaderを維持する。gfx1030はVGPR53→48、LDS0、spill0。証拠は`mxfp8-e4-decode-gfx1030-same-process-ab/`。
- r8は両target release build exit0、source before/after一致、H3契約31 test＋466 subtest PASS。gfx1201 FP8 M2..4のversion/revision/index policyとgfx1030 staged32 loaderを含む。公開staged32 testを既存sourceのcompile defineで追加し、1023／1024／1025・M1..4で既定ID93切替、独立oracle最大1 ULP、finite／repeat／cleanupをPASSした。公開FP8 testはgfx1201の両tuple・M1..5で全出力独立oracle最大0 ULP、repeat／cleanupをPASS。rocprofでMT16x16x32のDTVB1／0を各12回、M5の旧MT16x128x32を計6回確認し、新policyが公開executeから実dispatchされることを確認した。証拠は`r8-public-staged32-gfx1030/`と`r8-public-fp8-gfx1201/`。両GPUのr8 full-model単回測定を開始した。

- r8 flag未設定・MTP幅2・chunk1024・warmup0／measured1はV620-A **202.907／19.338 tok/s**、TTFT40.415秒／E2E46.982秒、R9700 **416.122／17.396 tok/s**、TTFT19.725秒／E2E27.026秒。両方128出力hashは前候補（V620 r6／R9700 r7）と一致、cleanup zero。正式中央値・目標達成は未完了。比較は`r8-comparison.json`。R9700の残るFP8とprefill費用を絞るためr8 profileを開始した。

- r8 R9700 profileは通常r8と出力hash一致・cleanup PASS。prefillのinclusive dispatch durationはlookahead約9.033秒、Q8/W16約6.206秒、decodeはID94約1.204秒、残るFP8 MT16x128x32約1.156秒／6968回、staged32 stage1約0.526秒。新FP8 MT16x16x32の2派生は約0.438秒／3744回と約0.334秒／4624回。重なりを含むため直列合計や通常性能へ換算しない。次の候補はQ8/W16 packed vec4 loaderと、既存policy対象外のFP8 shape。V620はloader改善で小Mの費用比が変わったため、MTP幅3をr8でも単回比較する。

- r8 V620-AのMTP幅3はprefill202.844／decode18.787 tok/s、採用86/126、HIP-only／cleanup PASS。staged32改善後も幅2の19.338を下回り、既定幅2を維持する。

- MTP full-commitの`resolve_decode_block`が返す出力cloneを省く案は、幅2で約30KiB／全採用blockのhost copy削減に留まり、GPU dispatch／replayを減らさないため当面保留する。AIによる任意改善案であり、適用範囲は全採用時のみ、費用はcore API／frontend分岐とfocused検証。現在の性能目標に対する優先度は低く、Phase83.5終了時に未採用なら再評価対象として残すだけで完了gateにしない。

- R9700 Q8/W16 packed vec4 loader scratchは5有効caseでfull control／repeat bit一致、独立oracle最大1 ULP、finite PASS、5境界外caseの両launcher拒否を確認した。200ms warmup＋20 AB／BAの長prefixは78.765→78.090ms／79.241→78.561msで約0.86%短縮、M129境界はほぼ同速。VGPR103→107と小さい利益を踏まえ、production統合は保留し、計算側のlane-parallel softmax候補を優先する。証拠は`mxfp8-q8w16-followup/packed-vec4-same-stream-gpu-r1.log`。

- r8 profile／packed vec4 probe終了後、R9700 serviceを既存production binary／設定で復旧し、healthz／readyz 200を再確認した。`service-state.json`のrestore_requiredはfalse。次のscratch準備はGPUを使わず、R9700の次回測定時は改めて使用状況を確認する。

- r8 R9700の残るFP8 groupはN5120のMT16x128が4824回／約0.740秒（K6144と17408はtraceだけでは分離不可）、K5120N17408が1072回／約0.231秒、K5120N12288が1072回／約0.186秒。これらと低費用のK5120N1024について、M1..4の5tupleを既存rank probeで比較する準備へ進む。exact MとK別の回数は推定で確定せず、profileのinclusive時間を通常性能へ換算しない。softmax候補は3queryの独立scalar更新をlane0/1/2へ分配し、同じquery内のFP32演算順を維持して2expfのSIMD実行を狙うscratchであり、採用は未決定。

- r8通常計時の残差はV620 decode6.567秒→目標6.350秒（127 decode token／20 tok/s）、約0.217秒／3.3%短縮。R9700はprefill19.687秒→16.384秒、decode7.301秒→5.080秒で約3.303秒／2.221秒の短縮が必要。profileのinclusive時間からこの残差を直接差し引かない。追加の局所候補としてstaged32の2expfをlane0/1へ分配し、同じFP32式を1回のSIMD expfへまとめるscratchを準備する。より大きいdecode残差には、partial accept時の全target replayをaccepted行のrecurrent／conv state復元へ置き換える最小設計も調査する。実装・採用条件の無断緩和は行わない。

- Q8/W16 lane-parallel softmax scratchの最初のGPU probeはcontrol/candidate差、次のprobeは両者bit一致になったがcontrol自身も独立oracle最大9 ULP／abs0.5で失敗した。2回の失敗後は新たなGPU反復を止め、candidate分配とscratch oracle／fixtureを分けて再計画する。許容差を緩めず、既存の検証済み公開Q8 probeを再利用して原因を切り分ける。production sourceへの反映は行っていない。

- 最初のQ8/W16 ILP案はdivergent branch内のshuffleを修正してcontrol/candidate bit一致となった。初期の広いE4値域fixtureでは既存control自身のoracle差9 ULP／abs0.5が残り、これを普遍的な数値PASSとは扱わない。既存公開fixtureに近い有限値域のr5では境界2caseで独立oracle最大1 ULP／abs0.0078125、全3caseでfullcontrol／repeat／finite PASSだったが、prefix8193／M128の200ms warmup＋20 AB／BAは19.343→20.984msで8.5%遅く棄却した。static v_expは6→2だがprivate segment16 B/lane、SGPR増が残った。次は一時score配列を消し、owner laneごとにmaximum／denominatorを保持して毎keyのmaximum broadcastも除く案へ限定して再計画する。失敗fixtureと全logは`qtile8-softmax-ilp/`に保持し、許容差変更で採用しない。

- staged32の2-lane expf案はV620-Aのprefix1023／1024／1025／8192／8193・M1/3、10caseでfullcontrol／repeat bit一致、finite、独立oracle最大1 ULP／abs0.00195312をPASSした。長prefixのAB/BA平均は8192 M1 0.726→0.733ms、M3 1.024→1.044ms、8193 M1 0.727→0.733ms、M3 1.021→1.024msで改善せず、gfx1030採用を棄却する。gfx1201はcompile-onlyで性能未検証。初回scratch起動のUUID誤記は実在V620-A `GPU-76a08c022586fed6`へ修正してから測定した。production変更なし、証拠は`staged32-softmax-pair/`。

- Q8/W16 owner-lane候補2はscore配列と毎keyのmaximum broadcastを除去し、gfx1030 SGPR61／VGPR102、gfx1201 SGPR62／VGPR99、両方private segment0／LDS2048／occupancy維持となった。V620-Bの4caseはfullcontrol／repeat bit一致、finite PASS。境界の独立oracle最大1 ULP、7168/M1024は3query×2heads×全256次元のsampled oracleで0 ULP。200ms warmup＋20 AB／BAは境界約4.7%、8193/M128約4.7%、実prefill相当7168/M1024は139.922→135.944msで約2.84%短縮した。gfx1030の統合候補として残し、gfx1201比較を準備する。証拠は`qtile8-softmax-ilp/gpu-run-candidate2-r2.log`。長caseを全出力独立oracle済みとは扱わず、初期large-range診断も保持する。

- R9700追加FP8 rank probeは5tuple×M1..4×10候補の200行を完了。全出力rank0比較／独立oracle最大0 ULP、finite／3 repeat一致。初回終了137と途中継続の意図的停止130は単独PASSではなく、完了行を保存して最終継続exit0まで結合した。M2..4のK5120/N12288→index123374、K5120/N1024→123373、K5120/N17408→123375、K6144/N5120→123373、K17408/N5120→123374を既存version/revision照合付きpolicyへ追加した。単体倍率約1.29〜2.96倍、M1は既存調整済みなので変更しない。r8 archiveの公開controlを7tuple×M1..5へ拡張し全出力oracle最大0 ULP／cleanup PASS。新policyの公開GPU確認はr9で行う。証拠は`fp8-smallm-gfx1201-rank-more/combined-root-rank-summary.json`と`r8-public-fp8-more-gfx1201-control/`。

- Q8/W16 owner-lane候補2はR9700でも4caseのfullcontrol／repeat一致、finite、長case sampled oracle最大0 ULP、exit0。7168/M1024は78.781→74.657ms（約5.2%短縮）、8193/M128は13.363→12.655ms。両targetへ同じ計算順のbodyを統合し、r9で公開GPUとfull-modelを検証する。単体の短縮率はモデル全体の値ではなく、large-range fixtureの既存control感度は未解決として保持する。証拠は`qtile8-softmax-ilp/gpu-run-candidate2-gfx1201-r1.log`。R9700 serviceは今回batch中停止中で、終了後に既存binary／設定へ復旧する。

- r9両target buildはexit0／source before-after一致、host runtimeとH3契約31 test＋466 subtestはPASS。公開FP8 7tuple×M1..5は全出力oracle0 ULP／repeat／cleanup PASS、traceで追加index123375に対応するMT16x16/WSGRB1も9dispatch確認した。一方、公開attention検査は両targetともprefix0/M128のQTile4で非有限値となりFAIL。Q8移植時にQTile4のVロードを512-thread用のelse節へ誤って変更し、256-threadのQTile4では実行されなくなっていた。QTile4の同じthreadがK/Vを読む旧bodyへ戻し、r9bを作って公開検証をやり直す。失敗r9でfull-model測定は行わない。既存境界検査で検出した統合不具合として記録し、許容差・受入条件は変えない。

- r9b両target buildはexit0／source before-after一致。公開attentionは両targetのQTile4/QTile8切替境界を含む既存matrixで独立oracle最大1 ULP、QTile4比較0差、repeat／cleanup PASSとなり、r9のロード漏れを解消した。FP8 sourceはr9から不変であり、r9の35case／trace証拠との対応を保存した。r9b両targetの8192/128・幅2・chunk1024・warmup0／measured1のモデル測定を開始した。

- r9b既定候補・幅2・chunk1024・warmup0／measured1はV620-A **205.114／19.335 tok/s**（prefill39.939秒／decode6.568秒）、R9700 **425.690／18.479 tok/s**（prefill19.244秒／decode6.873秒）。両方128tokenはr8と一致、HIP-only／fallbackなし／nonfinite0／cleanup zero、allocation peak24,713,604,048 B。単回探索であり正式中央値・目標達成とは扱わない。R9700残差はprefill約2.860秒、decode約1.793秒。比較は`r9b-comparison.json`。MTP M3 prefixのGDN checkpoint実装とQK-only WMMA scratchを並行して進める。R9700測定終了後は既存serviceを同じbinary／設定で復旧し、healthz／readyz 200を確認、restore_required=falseへ戻した。

### M3 prefix checkpointの統合（2026-09-10）

- nativeのM3 row0/row1 conv／recurrent保存、private prepare／validate／commit／discard、Rust backend橋渡し、Qwen部分採用時のreplay省略を実装した。公開C APIは117関数を維持し、4内部symbolだけをH3の識別対象へ追加した。全state検証と出力sliceを変更前に行い、復元後の失敗はrequestをpoisonする。非対応backendのprepareだけを明示Unsupportedへ接続し、既存replayへ戻す。
- 復元先はactiveなM3終了slot、inactiveな検証前slotは保持する。初期案のpre-block slot上書き、全modelへの一律3倍メモリ予約、誤った24 GDN層の判定は統合中に修正した。実model configは64層中48 GDN／16 Full Attention。層数を固定した採用判定は使わず、全stateのbackend capabilityを照合する。追加planeは初回prepare時に予約・確保し、state寿命まで再利用する。
- Qwen focused host testは70 PASS／1 ignored、native公開host回帰と両targetのkernel compileはPASS。これは新しいcheckpointのGPU証拠ではない。Qwen3.8 qk16／value48／d128／conv4の独立GDN oracle、非zero初期state、accept1／2／full3、次M1、再arm、stale／invalid row、rewindのpayloadを検証する`phase83_5_linear_checkpoint_gpu_test.cpp`と任意CMake入口を追加した。次buildで両target実機検証と8192/128を行う。
- QK-only WMMA scratchは直接MXFP8 decode、causal tail、非整列row、BF16 stagingがexactでないtileのscalar fallbackを実装し、gfx1201 compile-onlyを通過した。QK treeのN1根拠は未確定。実機はdiagnosticとして比較し、exit0やcompile成功だけで数値採用を認定しない。R9700 serviceは接続0を確認してこの検証batch中停止し、終了時に元へ戻す。

- r10 checkpoint候補は両target release build exit0、source before/after一致、H3契約31 test＋466 subtest PASS。新GPU testの初回strict compileでは独立oracleのV起点の範囲外アクセスを検出して修正し、同じruntime archiveへtestだけを再linkした。実機r2は両targetともcheckpoint無効のcontrol M3から独立oracleとの不一致（最大31728／31677 BF16 ULP、finite）でexit1。checkpointの正しさや性能の証拠にはならないためモデル測定へ進めず、oracleのQ/K/V配置と計算仕様を再照合する。失敗logは`r10-public-checkpoint-gfx1030-r2/`と`r10-public-checkpoint-gfx1201-r2/`へ保持する。

- r10 GPU oracleのK起点をQ幅1個分へ修正し、gfx1030のkey-major recurrent物理配置を独立oracleへ反映した。逐次oracleのconv historyは各row後に更新済みなので、batch kernelのrow offsetを重ねる修正案はroot確認で取り消した。runtimeは変更せずtest-only relinkしたr3は両target exit0。非zero初期state、M3 control／checkpoint、accept1／2後のstateと次M1、full3、再arm、stale／invalid row、pre-blockへのrewindをPASS。BF16出力oracle最大はgfx1030 0 ULP、gfx1201 3 ULP、conv payload bit一致、recurrent最大絶対差は約5.30e-7／5.46e-7。これは対象GDN fixtureの証拠で、full-model BF16品質同等性を意味しない。次にr10両targetの8192/128単回測定を開始した。
- QK-only BF16 WMMA scratchは実production Q8/W16 objectと同processで再比較した。前回scalar surrogateは実productionより遅く、採用判断の比較対象として不適切だった。実production比較の長prefix7168/M1024では77.439→201.127 ms（AB）、77.749→200.587 ms（BA）と約2.6倍遅い。既存bounded fixtureでは全出力bit一致／oracle一致／repeat PASSだが1.792→5.883 ms、wide fixtureはcontrol自身のoracle超過も残る。数値許容差を変えず不採用とし、N1成立も認定しない。試行と失敗を`qtile8-wmma-qk/gpu-run-actual-production-control.log`へ保持する。production変更なし。

- r10幅2／chunk1024／8192入力128出力／warmup0 measured1はV620-A **205.096／24.176 tok/s**、R9700 **425.023／22.893 tok/s**。decode wallは6.568→5.253秒／6.873→5.547秒で、両方128token hashとdraft採用数はr9bと一致。HIP-only／fallbackなし／nonfinite0／cleanup zero。追加request allocationは307,888,128 B（293.625 MiB）、session allocation総peak25,021,492,176 B。sysfsの実機VRAM peakはV620 30,158,536,704 B／R9700 30,284,169,216 Bで、library等を含む値とは区別する。測定中GTTはV620約15.06 MB／R9700約266.67 MBで起動時とほぼ同じ。target kernel dispatchは112546→87667／120051→87667へ減り、checkpointの追加確保と再計算削減が実モデルでも観測された。V620は単回で両目標超過だが正式中央値は未取得、R9700はprefill約2.890秒／decode約0.467秒の残差がある。Phase未完了。`r10-comparison.json`へ保存し、次にR9700 r10のprofileを開始した。

- r10 R9700 profileもexit0、通常測定とtoken hash一致、cleanup PASS。prefill inclusive kernel durationはID89 lookahead約9.199秒／1344回、Q8/W16約5.705秒／119回。decodeはID94約0.978秒／9072回、新FP8 MT16x16の2派生約0.524秒／3888回＋0.477秒／7776回、staged32 stage1約0.430秒／1002回。重なりを含むためwall残差へ直接換算しない。次の候補はllama.cpp RDNA4の128-thread tile配置を参考にした同Kahan順のtileM64/N64 lookahead（実M256／257／1024で比較）と、GDN checkpointの48層個別同期をまとめられるかの調査。後者のHIP API wall費用は未測定であり効果を確定しない。profile終了後、既存serviceを元のbinary／設定で復旧し、healthz／readyz 200を確認、`restore_required=false`へ戻した。

### 次の同期削減の実装範囲（2026-09-10）

- M3部分採用の48 state個別commit（96 D2D copy、48 stream fence）を、private batch commitの96 copy／1 fenceへまとめる。コピー量と採用plane、M3の演算・sampling規則は変えない。全handle／context／queue／重複／generation／rangeをコピー前に検証し、全コピーと最後のfence成功後だけ全state metadataを公開する。
- preflight失敗は未変更で返す。copy投入後の失敗はcontextをpoisonし、部分上書きからlegacy replayへ戻さない。既存single-state APIは1要素batchとして互換を維持する。installed C ABIを増やさず、private bridge／adapter／Qwen commit loopだけをつなぐ。
- 採用判断は複数stateのpayload／invalid入力／失敗処理と対象GPU検査、8192/128のactual wallとmemoryで行う。同期短縮の実測前に速度向上を認定しない。既存Phase受入条件は維持する。

- V620 r10／幅2／chunk1024／8192入力128出力の1 warmup＋3 measuredがexit0。正式中央値prefill **199.804517 tok/s**（MAD0.129321）、decode **24.091034 tok/s**（MAD0.034567）、TTFT41.025719秒、E2E46.297444秒。prefill測定3回は201.851757／199.675196／199.804517 tok/s。decodeは20以上だがprefillは200未満なので未達とし、丸め・単回最良値で置き換えない。4runの128token hash一致、HIP-only／fallbackなし／nonfinite0／cleanup zero。証拠は`v620a-r10-mtp-checkpoint-width2-formal-r1/`。残差はprefill41.000074秒→40.960秒、約40.1ms。公開CLI/APIの実際の自動chunk候補との対応も確認中。

- tileM64/N64 4-wave lookaheadはexact gfx1201の4case（wide M256、down M257、wide/down M1024）で全出力control一致、repeat／finite、64点独立long-double oracle最大0 ULP、exit0。M1024はwide6.802→7.889ms／down6.830→7.790ms（AB）、BAも同傾向で約14〜16%遅いため通常prefillへ不採用。down M257だけ約7.5%改善したが今回の8192入力の主要tileではない。LDS30720→20480 Bに対しVGPR228→250、spill0。inactive-LDS先読み案はVGPR256／3spillでcompile段階で棄却した。証拠は`nvfp4-prefill-m64-lookahead/`。実行前にR9700 service接続0を確認して停止し、probe終了後に既存serviceを復旧しhealthz／readyz 200、restore_required=falseを確認した。
- 公開chunk selectorのread-only確認では、32GiB級はtarget候補2048→512で、benchmarkの固定target1024は通常候補に含まれない。MTP companionは両方1024。従来のr10を含む1024結果は明示benchmark設定の証拠で、通常CLI/APIの実効chunk確認の代用にはならない。state容量もbenchmark9563に対し通常の該当API要求は8320。通常APIから同じfixtureの8192/128をV620-Aで実行し、選択chunk／placement／MTP／cleanupを確認する。最適化実装の呼出し経路は共通だが、性能達成を未確認の公開設定へ一般化しない。

- r10通常API V620-Aの8192/128（`p835-v620a-r10-default-coding-r1`）はchunk2048を選択したが、layer35のKV growでout-of-memoryとなりFAIL。実機VRAM peak33,995,030,528 B／total34,342,961,152 B。サーバー終了時のsession allocation／quarantineはzero、遅延後sysfsも起動前約17.2 MBへ復帰し、CPU fallbackは使っていない。正常な公開経路として合格にはしない。
- placement auditはmodel22.5 GB resident後もavailable34,311,503,872 Bを返しており、HIP bridgeがsession作成時の空き容量を保持していることを確認した。さらにincremental estimate約4.049 GBと実機peakの差があるため、native provider／libraryの保持scratchと見積もり対象を調査する。未検証のモデル名分岐で1024へ固定せず、通常経路でメモリに収まるchunk選択と実際の性能を確認する。

- HIP bridgeのavailable memoryをsession作成時のcacheから、要求時の`Context::query_device(device_index)`へ変更した。query失敗はNoneを返し、以前の空き容量を再利用しない。total capacityは不変情報としてcacheを維持し、host-only adapter fixtureの固定値はtest専用variantへ分離した。`cargo check -p sllm-hip --lib` PASS。実機の見積もり差・OOM解消は次候補で確認する。
- tileM128/N64を維持しStageK64→32へ変更したlookaheadはgfx1201の4caseで全出力control一致、repeat／finite、64点long-double oracle最大0 ULP、exit0。wide M256は約3%、down M257は約11%改善した一方、主要M1024はwide6.882→6.979ms、down6.783→7.797ms（AB）、BAも改善せず通常prefillへ不採用。LDS30720→15360 B／VGPR228→186／spill0だけでは速度改善を保証しない。証拠は`nvfp4-prefill-k32-lookahead/`。これ以上tileだけの派生を増やす前に、公開経路でのmemory／chunk問題とprovider workspaceを優先する。

- live-memory query変更後のHIP bridge host testsは23 PASS／0 failed。queryの実GPU容量反映は次buildで確認する。K32 probe後のR9700既存serviceもhealthz／readyz 200を確認し、restore_required=falseへ戻した。

- private batch commitをnative／HIP bridge／core adapter／Qwenへ統合し、単一stateは共通helperの1要素batchとして維持した。複数admission lockはstate ID順で取得し、逆順batch間のdeadlockを避ける。Qwen対象テストは3 PASS（複数stateへの1 batch、非対応replay、失敗poison）。native host testの追加関数に失敗時cleanupの成功を返す誤りがあり、失敗は必ずfalseへ修正した。post-copy／final-fenceのfault注入ではlength／generation／active slotが未公開のままでcontextがpoisonすることを子processで確認し、host ctest1/1 PASS。
- native prepared matmul workspaceはplanごとにdirect hipMallocで確保し、plan寿命まで保持され、graph workspace／Rust allocation ledgerに含まれていない。失敗runの実機peakとtracked model/request/workspaceの差は約8.379 GBで、全部をKVの費用とは扱わない。live queryによりmodel＋MTP resident後のfreeは反映できるが、まだ作成していないprovider scratchは別途扱う必要がある。共通scratchの再利用とそのfootprintの報告方法を調査する。
- batch commit＋live-memory queryをsource freezeし、manifest同期後にr11両target release buildとH3契約検証を開始した。公開memory問題の解消はまだ未確認であり、1024 benchmarkの速度と通常APIのメモリ収容を分けて追跡する。

- r11両target release buildはexit0／source before-after一致、H3契約31 test＋466 subtest PASS。fresh archiveへlinkした公開checkpoint GPU testは両targetで3state batchのpreflight失敗時payload／metadata不変、row1復元、各stateの次M1、既存single／rewindをPASS。batch／次M1はBF16 oracle0 ULP、conv bit一致、recurrent最大絶対差4.66e-10。全test範囲の上限は既存再armを含めgfx1201 3 ULPのまま。続いて同じ1024 benchmark条件で両GPUのr11単回比較を開始した。

- r11 batch commit＋live query、幅2／chunk1024／8192/128単回はV620 **202.485721／24.144791 tok/s**、R9700 **420.974114／23.060314 tok/s**。両方128tokenはr10と一致、HIP-only／fallbackなし／nonfinite0／cleanup zero、session allocation peak25,021,492,176 B。R9700 decode wall5.547464→5.507297秒で差は約40ms、V620はほぼ同水準であり大きな速度向上とは主張しない。単回のばらつきと区別する正式比較は未実施。`r11-comparison.json`へ保存。通常APIのOOMは未解消確認のまま、provider workspaceを優先する。R9700測定後は既存service復旧を開始した。

未完了。完了または中止時にarchiveへ移し、試した変更・測定結果・採否を対応するhistoryへ残す。

- r11後の既存R9700 service復旧を実状態で確認し、healthz／readyzとも200、restore_required=falseを記録した。live free-memory queryだけで通常経路のOOMが解消したとは認定せず、r11 V620の通常API・context8320・8192/128 coding-only検査を開始した。
- gfx1201 ID92 arithmetic移植scratchのcompile／linkは成功したが、root確認で比較対照が現行tuned index123374ではなくrank0、計時も要求した200ms warmup／20組交互AB/BAではなかった。GPU実行前に比較対照・計時・oracle判定を修正する。既存selectorがgfx1030限定という理由だけではgfx1201での試行を棄却しない。

- r11通常V620 API（8192/128、context8320、target chunk2048）は同じlayer35.kv_appendの物理KV commitmentでOOM、検査FAIL。live queryによりplacement availableは11,131,682,816 Bへ更新されたが、incremental estimate4,049,263,001 Bでadmitし、physical peak34,311,659,520／34,342,961,152 Bへ到達した。空き容量cacheの修正だけでは解消しないと実証した。HIP-only／fallbackなし、終了時ledger全0／quarantine0。詳細は`.local-artifacts/phase83/p835-v620a-r11-default-coding-r1/`。native prepared matmul scratchの再利用と事前見積もりを次の修正対象とする。

### Queue所有の低精度matmul作業領域（2026-09-10・実装前の範囲）

- 最初の修正は既存lowp providerのM>1に限定する。NVFP4／MX系のchecked workspace計算を維持し、planごとのhipMallocをqueue所有の再利用領域へ移す。model名やchunk1024固定では選別しない。M1とnative FP8 Ltの永続scale pointer、既存graph captureの範囲は維持する。
- 同じqueueではquantizeとmatmulのenqueue全体をmutexで保護し、異なるqueueは独立した領域を持つ。必要容量が増えた場合は幾何的に拡張した別領域を確保し、以前の領域をqueue解放まで保持する。実行中pointerの変更・解放は行わない。
- allocation成功／free成功時だけnative current／high-waterを更新し、free失敗は既存poison／quarantineに従う。native診断値の追加だけでRust allocation ledgerへ反映済みとは表記しない。admission側の見積もり反映は続けて行う。
- 受入は同queue再利用・拡張、異queue独立、失敗時の事前rollback、解放をfocused host検査し、変更対象の両GPU公開matmul数値検査と通常API8192/128で確認する。演算式・量子化recipeは変更しない。速度目標・正式反復・公開lifecycleというPhase全体の条件は維持する。

- gfx1201 ID92-body M3/K5120/N10240 scratch r3は現行Lt index123374（version100401／revisioncd957402）との全30,720出力一致、独立oracle最大0 ULP、finite／repeat、exit0を確認した。一方、20組AB/BAの単発計時はABがcontrol約44／candidate約62 us、BAがcontrol約113／candidate約94 usと順序依存が大きく、全体中央値112.561→94.161 usを改善率として採用しない。各event内32連続launchへ変更して再比較する。証拠は`fp8-smallm-gfx1201-custom-r3-gpu/`。

- 同FP8 scratch r4は各event32連続launch、20組AB/BAで再測定し、全出力oracle／control0 ULP、repeat／finite、exit0を維持した。control→candidate平均はAB39.209→87.312 us、BA40.653→88.307 us、全体39.931→87.810 us（candidate約2.20倍遅い）。gfx1201でのID92-body採用を棄却し、現行tuned Ltを維持する。r3の単発中央値による誤った改善判定は行っていない。証拠は`fp8-smallm-gfx1201-custom-r4-gpu/`。GPU終了後に既存R9700 serviceの復旧を開始した。

- FP8 scratch batch後の既存R9700 service復旧はhealthz／readyzとも200を確認し、restore_required=false。V620 r11 OOM終了後も遅延sysfs確認でVRAM17,215,488 B／GTT15,020,032 B／busy0の基準値へ復帰した。

- queue poolのmatmul側を実装し、M>1のNVFP4／MX workspaceを実行時のqueue領域へ切り替えた。rootの統合確認でallocation失敗時のcleanup前unlock漏れ、helper内部で同じmutexを再取得する箇所、vector拡張失敗後のdevice allocation計上漏れを見つけ、順次修正した。vector容量をdevice確保前にreserveする構造へ変更し、成功したdevice allocationが未計上になる経路を除去する。ホスト回帰とGPU検証は未完了。Rust format検査はPASS。

- r12 queue-pool候補はrootがhost-only unused警告とfake実行許可を修正し、focused host runtime ctest1/1 PASS。新M1検査はprepare時の個別確保を確認し、M1実行は既存lifetime testの範囲とする。両target release build exit0／source before-after一致、H3契約31 test＋466 subtest PASS。V620／R9700の公開NVFP4 small-Mは両tuple×M2/3/4の全出力oracle最大0 ULP、prefillは既存非整列境界matrixで5×5 sampled oracle／全出力finite・repeat、cleanupをPASSした。両GPUの通常API8192/128（context8320、自動chunk）を開始した。gfx1030 binaryは`r12-queue-scratch/`、gfx1201は`candidate-gfx1201-r12-queue-scratch/`へ固定している。

- r12通常APIは両GPUで8192入力／128公開出力、chunk2048、context8320、MTP有効、HTTP/SSE200、HIP-only／fallbackなし、終了ledger0／quarantine0でPASS。V620 E2E46.160秒／first-content40.261秒、R970025.254秒／19.697秒、MTP採用74/107／76/104。physical peak32,639,619,072／32,765,272,064 B、GTTは基準近傍。r11と同じ通常経路でOOMを解消した。単回のSSE計時から正式prefill/decode達成は認定せず、区間別反復を続ける。比較は`r12-default-api-comparison.json`。残るnative FP8等の個別確保も含むallocation-free footprint queryをbackend層へ実装し、admission見積もりへ反映する。

- r12通常API batch終了後、既存R9700 serviceを同じbinary／FP16 KV設定で復旧し、healthz／readyz200、restore_required=falseを確認した。

### Native matmulの事前見積もり（2026-09-10・実装中）

- unbound semantic descriptorからbackendへ問い合わせ、plan永続bytes／queue再利用bytes／context再利用bytesを区別する。HIP側はprepareと同じchecked layout計算を共有し、device allocationを行わない。未対応・overflow・不明を0 bytesへ置換しない。
- Qwen側はplanの生存するrow variantを計上し、queue領域は逐次演算ごとに加算せず、最大要求と保持済み拡張領域の上限で見積もる。幾何拡張の保持合計は最終capacityの2倍未満だが、最大要求の2倍未満とは限らない（要求2→3→5で確保2+4+8=14）。事前の一般上限には最大要求の4倍未満を使い、target／MTPの別queueを区別する。
- FP8・M1の個別確保は診断表示だけでなくadmissionへ加える。native matmulの見積もり追加を、全native資源の実測ledgerが完成したという意味には扱わない。
- coreへ`PreparedMatmulFootprint`とsession/adapter queryを追加し、unknownを`None`で保持する。core library checkはPASS。HIP loweringとQwen/public callerの接続は未完了。

- r13 native queryとprepareのshared checked layout、HIP unbound descriptor lowering、CLI/APIのcandidate admission接続を実装した。plan永続sum／queue最大要求の4倍上限／context最大を分けてauditへ出す。NVFP4 M1/M2/M3、MXFP4/8/6、FP8 outerの既知bytes、queryのallocation／launch／device-query不変、invalid／overflow／unsupportedで出力未変更をhost runtime ctest1/1でPASS。HIP lowering9 test、CLI/server focused testとcrate checkもPASS。core集計のfocused testと統合確認は進行中。
- benchmarkへ明示state capacityを追加し、既存9563の測定を保持しながら通常APIと同じ8320を選べるようにした。target／companion graphとreportに同じ設定を使い、選択した入力＋出力を収容できない設定を拒否する。benchmark crate checkはPASS。次回性能比較はchunkとstate capacityの両方を記録する。

- core footprint集計のfocused test3件（row重複排除、plan sum／context max／queue保持上限、unknown／backend error、invalid row／overflow）がPASS。H3は31 test＋466 subtest PASS。format後に両targetのr13 release buildを開始し、MXFP4/8/6用の既存evidence binaryも同じsource identityでビルドする。
- gfx1201 N128/16-wave scratchはM2048／M1024×K5120,N17408／K17408,N5120の全4caseでfinite／repeat、全出力control bit一致、sampled oracle最大0 ULP、exit0。20組AB/BAの両順序とも現行N64より約14〜18%遅く、採用を棄却した。M2048 wideはcontrol13.359/12.946 msに対しcandidate15.477/14.946 ms。N128/8-waveはcompile時85 VGPR spillで未実測棄却、16-wave化でspill0にしても速度改善へつながらなかった。証拠は`prefill-next-r12/gpu-r9700-n128-w16-r1/`。既存R9700 serviceは同じunit／launcherで復旧しhealthz／readyz200、restore_required=false。

- r13両target release buildはexit0、source before/after一致。固定candidateにserver／CLI／benchmark／MXFP4・MXFP8/6 evidenceを保存した。gfx1030公開NVFP4 small-M全出力oracle最大0 ULP、prefill非整列境界sampled oracle最大0 ULP／全finite・repeat／cleanupをPASS。MXFP8/6は各M1/3/17（最大relative error約0.00383）、MXFP4はsynthetic M1/3/7と実重みM1/3/7（最大約0.00390）で既存独立oracle／HIP-only／cleanup0をPASS。両GPUの通常API見積もり反映後の回帰は実行中。
- r13の統合確認ではquery・bridge・target admissionにcorrectness blockerは見つからなかった。scheduleを持たないcore provisioningの純粋layout見積もりは基礎分のままとし、公開requestのschedule-aware判定と区別する。serverではMTP residentを先に確保した後のlive availableを使うが、後から作るcompanion requestのnative領域はtarget queryへ未合算。CLIではcompanion residentを後から確保する経路もあり、target＋companion全体の厳密なreservationは未完成。既存safety reserveや今回の完走を、全configurationのtight-memory保証とは呼ばない。

- r13 V620通常APIは8192/128、chunk2048、context8320、HIP-only、HTTP/SSE200、cleanup0でPASS。prepared plan2,789,652,440 B／queue上限80,216,064 B／context0を計上し、incremental required6,919,131,505 Bとlive available11,131,682,816 Bで採用した。E2E46.436秒、first-content40.960秒、physical peak32,658,235,392 B、GTT基準近傍。r12と生成textが一致、MTP74/107採用。通常shapeと揃えた明示benchmark（chunk2048／state8320、1 warmup＋3 measured）を開始した。正式結果は未確定。

### 次の性能実験: MTP draft M1の既存graph span利用

- r10ではtarget／draftともgraph replay／span／capture node数が0。現行completion selectorがMTPを除外し、target検証M3もRust/nativeのM1限定capture条件から外れる。wall/GPU timestamp差だけで原因を断定しないが、launch／completion処理を減らす具体的候補としてdraft M1を先に試す。
- 既存のrequest専用queue、M1 stateless `layer.*` graph spanとdeferred completionをcompanionのM1へ限定して使う。MTP prefix primingのM>1、copy、KV/state操作、terminal sampling／argmaxは既存のeager境界を保つ。M3 contract拡張と新しい汎用graph抽象化はこの実験に含めない。
- 受入はscope／rollbackのfocused host検査、同一候補でoff/onのdispatch／span／replay実績、数値・token差の確認、cleanup／cancel、公開経路と速度の比較。速度目標やBF16品質に関するPhase全体の条件は変更しない。改善が確認できれば既定へ反映し、悪化・未選択なら理由を記録して棄却する。
- pinned llama.cppのubatch shape/inputが一致するgraph予約・再利用を参照する。これはggml graph-bufferの再利用でありHIP Graph replayそのものの証拠ではないため、M1とM3を同一captureへ流用する根拠にはしない。

- r13 R9700も公開NVFP4 small-M／prefill、MXFP4/8/6の既存数値matrixがPASS。通常API8192/128はchunk2048×4、HIP-only、HTTP/SSE200、cleanup0、MTP76/104採用でPASS。native plan2,789,652,440 B／queue上限80,216,064 B／context0、incremental6,912,420,619 Bをlive available10,932,453,376 Bと比較した。E2E24.806秒、first-content19.249秒、physical peak32,765,296,640 B、GTT基準近傍、r12生成textと一致。証拠は`r13-gfx1201-validation-r1/`と`r13-default-api-comparison.json`。既存serviceは同一設定で復旧しhealthz／readyz200、restore_required=false。復旧開始後の追加formal runは行わず次batchへ残す。
- r13 V620の通常shape一致benchmarkは1 warmup＋3 measured、8192/128、MTP幅2、chunk2048／state8320でexit0／cleanup0。prefill中央値199.614164592 tok/s（MAD0.577756817）、decode23.834053828（MAD0.035897350）、TTFT41,069.559 ms、E2E46,398.130 ms。prefillの3値は200.191921／199.614165／197.044038であり200未満の中央値を丸めてPASSにしない。warmup201.013743も達成判定には使わない。全反復128token hash一致、MTP74/107採用、HIP-only。target graph span128／capture node1128だがreplay0で、M1が一度だけ生じる終端部分ではcaptureを再利用できていない。証拠は`r13-v620-formal-comparison.json`。Phase全体は引き続き未完了。

### 次のprefill実験: Q8/W16のK/V 4行staging

- 現行MXFP8 Q8/W16はkey positionごとのLDS loadとblock同期を行う。4つの連続keyを一度にLDSへloadし、barrierを共有してから既存online softmax／V積算をkey昇順のまま4回実行するscratch候補を試す。Q8／16wave／512thread、query mapping、FP32演算・expf・BF16丸め、selector scopeは維持する。
- LDSは約2 KiBから8 KiBへ増える。同期削減の利益とVGPR／occupancy／bank conflictの影響は未測定。最初はB=4だけとし、B=8やQK WMMAの再試行へ拡大しない。端の非整列keyとcausal maskを保持する。
- pinned llama.cpp `fattn-tile.cuh` の複数K/V行stagingを参照するが、その数値式はコピーせず現行sLLMの順序を維持する。受入はresource compile、現行production providerとの同一process AB/BA比較、独立oracle／full finite・repeat・control一致、実際のchunk2048範囲での改善。失敗時は理由を記録して候補を棄却する。

- r14 draft M1候補を実装し、gfx1030／gfx1201別の明示opt-inを追加した。generic MTP除外は維持し、companion専用queueを用意した上で実行時token_count==1だけdeferred completionを使う。M>1 prefix、M3 target、MTP KV chainへ拡大しない。focused selector2 testとQwen38関連24 test（既存3 ignored）はPASS。benchmarkへdraft側のspan／capture／replay／segment／fence数を加え、両target build exit0、source before/after一致を確認。同じr14 binaryでoff/onの実機比較を開始した。
- companionまで含む全configurationの厳密なreservationはr13統合確認由来のnonblocking追加改善候補。費用はtarget／companionの同時scheduleと共有resident控除をpublic callersへ渡す追加実装で、未見積もり。現在の固定8192/128・32GB収容と混同せず、Phase83.5 closeout時に後続へ引継ぐか再評価する。新しい完了gateにはしない。

- r14 V620 draft-M1有効側は8192/128、chunk2048／state8320、単回でprefill203.282／decode24.054 tok/s、exit0／HIP-only／cleanup0。draft graph span2／capture kernel13／replay212を実行し、draft segment304／physical fence296。生成128token hashはr13 formalと一致し、MTP74/107採用。これは同一候補off/on比較の片側で、まだ採用・正式達成判定ではない。続いて同じr14 binaryのoff側を計測する。

- r14同一binaryのoff/on単回比較は両GPUで128token hash一致、HIP-only、cleanup0。V620 decode5,329.205→5,279.764 ms（23.831→24.054 tok/s）、R97005,463.051→5,437.816 ms（23.247→23.355）。draft replay212／206、span各2、capture kernel各13。物理fence counterはoff45／42→on296／284で、profiled completion queryとdeferred queue fenceの計上範囲が違うため「同期全体が減った」とは主張しない。約0.5〜0.9%の単回差では既定採用を確定せず、M1候補は評価中とする。証拠は`r14-mtp-draft-comparison.json`。

### 次のdecode実験: 固定MTP検証M3のstateless graph span

- M1 draftだけではR9700 decode目標へ不足するため、固定幅2 MTPのtarget検証M3へ既存stateless spanを拡張する。M1／M3は別のrequest-local replay stateとし、row数が異なるgraphを共有しない。通常3-row prefillや任意decodeへは有効化せず、MTP検証routeを明示して限定する。
- native plan resolverはM1／M3と同一span内row一致を検証する。projection-pack／GDN bundleはM1限定を維持し、M3ではeagerにする。KV append、attention、GDN state、copy、terminal selectorはcaptureへ入れない。
- M3 NVFP4等のqueue workspaceはwarmupで既に確保済みであることをcapture前に確認し、capture中に新規確保しない。capture対象のplan／buffer／queue lifetimeは既存pinを維持する。replay可能な連続spanが0なら新候補はeagerのまま継続し、正常な要求を失敗させない。
- 受入はM1/M3分離・mixed-row拒否・cold workspace拒否のfocused host検査、両GPUのM3 direct-vs-replay数値・lifetime検査、固定8192/128 off/onの出力／state／cleanupと速度比較。演算式・量子化・state公開順を維持する。新しいM3 native契約の検証であり、全row／全model capture対応の宣言ではない。

- B4 K/V stagingはscratchで両target compile/link PASS。gfx1201はcontrol99 VGPR／62 SGPR／occupancy12／LDS2048 B→candidate102／74／12／8192 B、gfx1030は102／61／9／2048→106／73／9／8192。spillなし。rootはproduction由来control、stagingのkey昇順とbarrier、runnerの実際の4caseを確認し、計時のAB／BA各中央値も出すようrunnerを補強してからGPU比較する。
- M3用公開NVFP4 probeは既存small-M testへ明示compile flagで追加した。eager warmup後の同じqueueでcaptureし、3行の正・負・1.5入力を変えてreplay、全出力独立oracleと解放を確認する。strict host syntaxはPASS、native M3実装とGPU実行は未完了。

- B4 K/V stagingの実GPU比較はV620の4caseすべてで約0.65〜0.80%改善し、AB／BA両順序で一致した。全出力control bit一致、repeat／finite、独立oracleは境界最大1 ULP／long 0 ULP。R9700は全caseで約1.7〜4.0%悪化したため棄却した。productionはgfx1030だけを測定済みB4へ変更し、gfx1201は元の1行stagingを維持する。selector／symbol／launch geometry／数値順序は維持。production compile-onlyは両target PASS、spill0。full-model改善はまだ未確認。証拠は`qtile-kv4-r14/`と同directoryの`production-adaptation.md`。
- M3 graph spanをRust/nativeへ実装した。request内M1/M3 replay stateを分離し、明示opt-inと固定MTP検証routeだけでM3を選択する。nativeはspan内row一致とwarm済みqueue scratchをcapture前に確認する。focused Rust Qwen38 25 PASS／既存3 ignored、graph replay 2 PASS、native host runtimeと両target strict HIP syntax PASS。queue pool参照はregistry／accounting lockとcapture排他で保護されることをfocused確認した。benchmarkへM3 selector設定を記録し、両targetのr15 buildを開始。GPU correctness／off-on比較は未完了で、既定採用は未決定。

- r15両target release buildはexit0／source before-after一致、H3 31 test＋466 subtestとRust format／diff検査PASS。V620公開NVFP4 M3 graphは両tupleでcapture各2 kernel node、入力3patternの全出力oracle最大0 ULP、release PASS。gfx1030 production B4の公開MXFP8 prefillは13境界caseで独立oracle最大1 ULP、既存Q4 controlとの全出力一致、dispatch／cleanup PASS（V620Bで実行）。
- r15 V620 M3有効の単回8192/128はprefill204.126／decode23.791 tok/s、128token hashはr13と一致、MTP74/107、HIP-only／cleanup0。target graph replay9,568／span312／capture kernel1,896を確認したが、decode改善とは未判定。CPU decodeは513 USER_HZ ticksでr14無効147より増えており、同じr15 binaryの無効側測定とreplay処理の調査を進める。

- r15 V620同一binaryのM3無効／有効はdecode5,329.869／5,338.237 ms（23.828／23.791 tok/s）、CPU149／513 ticks、同一128token／MTP74/107。単回で速度改善を示さず、M3は既定に採用しない。無効側prefill202.861 tok/sを確認したため、B4追加後の既定構成（draft/M3 graphとも無効）で1 warmup＋3 measuredの正式検査を開始した。証拠は`r15-v620-m3-comparison.json`、正式結果は未確定。

- r15 R9700公開M3 graphも両tuple／入力3pattern／全出力oracle最大0 ULP、release PASS。同一binaryの無効／有効はprefill429.984／425.574、decode23.410／23.515 tok/s、E2E24,512.797／24,688.568 ms。target replay0→9,384、span0→184、capture kernel0→768、fence各952、CPU decode284→547 ticks。同一128token hash、HIP-only／cleanup0。約0.45%の単回decode差と全体悪化では既定採用せず、現候補は評価を保留する。既存FP16 serviceを同じ設定で復旧しhealthz／readyz200、restore_required=false。証拠は`r15-r9700-m3-graph-r1/batch-summary.json`。
- 次候補調査では既存ID89内のtile派生に新たな有力案は見つからなかった。pinned llama.cpp `mmq.cu`のshared gate/up activation量子化を参考に、既存M1 projection-pairのM>1分解で重複する量子化を共有する余地がある。r10 quantizer全体0.634秒から削減上限はその一部（粗い上限約0.3秒）で、これ単独でR9700目標へ達するとは見込まない。現時点では調査候補で、実装・速度改善は未確認。M3 CPU増加もcapture／instantiate、replay、completion処理の費用をまだ分離できておらず、finalize batch化が主因解消になるとは断定しない。

- r15 V620既定構成の正式1 warmup＋3 measuredはexit0／HIP-only／cleanup0、全128token hash一致。prefill中央値 **198.577009079 tok/s**（MAD0.376165603）、decode **23.795903167**（MAD0.022624105）、TTFT41,290.900 ms、E2E46,619.376 ms。prefill3値200.720147／198.577009／198.200843で、200未満の中央値をPASSにしない。scratchのB4微改善からfull-model達成は導けず、同条件の無変更再測定は行わない。証拠は`r15-v620-formal-summary.json`。両GPUの速度目標は引き続き未達分がありPhase83.5を継続する。

- 次の費用分解として既存r15 binaryを変更せず、V620のM3有効runへ対象PID限定bpftrace uprobeを接続する。`sllm_graph_span_create`／`hipGraphInstantiate`、`sllm_graph_span_execute`／`hipGraphLaunch`、`sllm_completion_finalize_after`の回数とentry-return時間を集計し、finalizeはcapture内外を区別する。計測介入があるため速度達成・正式性能の証拠には使わない。既存off/onのCPU ticksだけからfinalizeを主因と決めつけず、全capture費用とsteady replay費用を分けて次の変更を選ぶ。

- r15 uprobeは対象PIDへのattachとbenchmark完走を確認し、create312回／39.389 ms（うちinstantiate20.735 ms）、execute9,568回／69.518 ms（うちlaunch41.009 ms）、finalize35,798回／45.768 ms（capture内1.676 ms／外44.092 ms）。包含関係のある時間を加算せず、これらではCPU増加約3.6秒を説明できないと判断した。finalize batch化を主因対策として先行実装しない。次に同じr15 binaryへ対象PID限定のwait／hipEventQuery計測とCPU stack samplingを接続し、初回graph create以前／以後を区別する。計測介入付きの速度は受入値にしない。

### 次のprefill実験: projection pairの入力量子化共有

- 既存NVFP4 projection pairをM>=64のprefillへ拡張し、現在別々のgate／upが重複実行する同一入力の量子化を1回へまとめる。既存gfx1030 ID87／gfx1201 ID89の演算順、入力scale、出力、2 projectionのenqueue順を維持する。FP8 GDN、M2〜4のID94、M1 graph contractはこの最初の実験へ含めない。
- nativeはM-aware checked workspaceとdispatch gridを使用し、M>1の共通量子化領域は既存queue poolへ置く。量子化から両projectionまでscratch mutexを保持する。M1のplan-owned領域と既存ABI値／symbolは維持する。元のlogical matmul 2個によるadmissionのqueue最大要求がpair要求以上であることを確認する。
- 初期の公開routingは明示opt-inで比較し、対象外row／roleは既存分解を維持する。未対応providerを無条件に別演算へ置換せず、候補の失敗を数値PASSにしない。受入はhostのoverflow／row／workspace lifecycle、両GPUの非整列rowを含むpair対独立2matmulと数値oracle、dispatch量子化1回、finite／repeat／cleanup、8192/128の通常shape off/onとmemory比較。Phase全体の正式速度・公開lifecycle条件は変更しない。
- pinned llama.cpp `mmq.cu:191-235`はMoEのexpert broadcastに対するtoken単位のquantize-once／scatterであり、dense gate/up pair共有の直接実装例とは扱わない。同じ入力の重複量子化を省く設計参考とし、sLLMの既存pairを拡張する。直接copyは予定していない。

- M3有効CPU stack samplingではdecode近傍のsample cyclesの68.21%がROCr `Runtime::AsyncEventsLoop`、1.72%が`BusyWaitSignal::WaitRelaxed`へ集中した。初回graph create以後の`hipEventQuery`は203,527回／294.215 ms、completion waitは33,135回／4.750秒。process CPUにはROCr helper threadも含まれるため、主queue fence数の一致だけで待機CPU費用まで同一とは言えない。perfのMONOTONIC_RAWとbpfのMONOTONICは約115.284秒の差を補正し、decode窓は近似として扱う。これを正式性能や厳密なCPU時間内訳へ昇格させない。証拠は`v620a-r15-m3-cpu-stack-r1/attribution-summary.json`。

### 並行scratch実験: gfx1201 Q12／24-wave prefill

- 既存Q8／16-waveは1 waveあたり3 queryを担当する。query tileを12、wave数を24、thread数を768へ同時に拡大し、1 waveあたりのquery数・register上の演算順・key昇順・owner softmaxを保ったまま、同一K/V行を共有するquery数を増やすscratchを試す。現行gfx1201の1行K/V stagingを基準とし、棄却済みB4 stagingと組み合わせない。
- pinned llama.cppのquery/KV tilingを設計参考とする。これはsLLMの現行bodyを使う独立候補であり、上流に同じQ12／24-wave実装があるとは主張しない。24-wave workgroupによるresident group数低下・tail overheadで悪化する可能性を含めて測定する。
- productionへ先に入れず、resource compileと同一process AB／BA各20組で比較する。prefix2048/M2048、1025/M129、8193/M128、6144/M2048を含む既存B4の境界・long matrixを再利用し、全finite／repeat／control一致と独立oracle、exact gfx1201とcleanupを確認する。改善を確認できない場合は理由を記録して棄却する。既存R9700 serviceはbatch後に同じ設定で復旧する。

- ROCrの[現行runtime source](https://raw.githubusercontent.com/ROCm/rocm-systems/develop/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp)／[signal source](https://raw.githubusercontent.com/ROCm/rocm-systems/develop/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/signal.cpp)には、監視signalの条件によりactive pollingを使う経路がある。ただしinstalled1.21.0のexact source対応とM3 signalがその分岐を通ることは未証明であり、原因の確定とは扱わない。supportedなgraph単位wait-mode切替は見つからず、全体のROCr設定変更や追加signal追跡へ調査を拡大しない。r14/r15 graph候補は採用を見送り、実験経路の整理時に費用・不採用理由を残す。性能改善の優先をshared quantizationとquery tilingへ戻す。
- prefill pairのRust opt-in routingを実装し、M>=64かつ既存NVFP4 pairだけを選択、M1の既存pairとM2〜63／FP8 GDNの既存分解を維持する。rootのQwen38関連host検査は28 PASS／既存3 ignored。native実装とGPU検査は未完了で、既定採用はまだ行わない。

- Q12／24-wave gfx1201 scratchはresource99 VGPR／62 SGPR／occupancy12／LDS2048 B／spill0でcontrolと同じ。全4caseのcontrol bit一致・repeat・finite、M129 full oracle最大1 ULP／long sampled0 ULPをPASSしたが、candidate/control比は2048/M2048=1.004、1025/M129=1.056、8193/M128=1.096、6144/M2048=1.023とすべて悪化し棄却した。Q16等へ同じtile派生を追加せず、量子化共有へ進む。証拠は`qtile12-w24-r16/batch-summary.json`。既存R9700 serviceは元のFP16設定で復旧しhealthz／readyz200。
- shared quantizationのnativeはM-aware overflow／dispatch gridとqueue pool lifetimeを実装し、hostのM63拒否／M64,65,127,128,129,512／両provider／pool growth・reuse、両target strict HIP syntaxをPASS。r16両target release buildはexit0／source before-after一致、H3 31 test＋466 subtest PASS。新しい公開pair数値probeとsame-binary off/onを開始する。

- r16両GPUの新prefill pair公開probeはexit1でFAIL。旧M1／FP8検査のPASSを新prefillの成功へ読み替えず、full-model off/onは開始しなかった。rootがnative executeのscale開始offsetにM乗算が欠け、`ceil(K/2)`の位置からscaleを書いてM>1のpacked activationと重なる不具合を特定した。`M*ceil(K/2)`へ修正し、layoutの回帰と失敗時のrow／member／metadata／oracle診断を追加して再buildする。証拠は`r16-public-prefill-pair-gfx1030/`と`r16-r9700-prefill-shared-r1/`。R9700 serviceは同一設定で復旧しhealthz／readyz200。

### 次のmatmul scratch候補: exact fixed-point外側累積

- ID89のK16 WMMA dotはE2M1のquarter-integer和で、`Q=4*dot`は整数かつabs<=2304。finite E4 scaleは`S/512`（abs(S)<=229376）なので、各termは`Q*S_a*S_w/2^20`となりFP32でexact。K=17408までの全fixed sum絶対値上限は約1.319e17でsigned64に収まる。
- この有限値scopeでFP32 Kahan外側累積をexact signed64へ置き換えるscratch候補を調べる。最終のFP32変換／outer tensor scale／BF16丸めは維持する。理論上は累積誤差boundを増やさないN1候補だが、BF16 tieで既存Kahanとの差が生じ得るため測定・記録する。NaN scaleを整数へ未定義変換せず、明示的なnonfinite出力経路を持たせる。finite入力以外のbitwise互換は未認定。
- 既存llama.cpp由来E2M1 signed-byte mappingを設計参考として、integer WMMAによるexact Qとfixed累積を検討する。上流に同じfixed累積があるとは主張しない。まずscratchでresource compileし、現行ID89との同一process AB／BA、独立高精度oracle、non-aligned rows／repeat／finite／nonfinite入力の挙動で評価する。64bit整数演算やscale decodeの費用が上回れば棄却し、productionへ先に導入しない。

- r16bでpacked activationのchecked `M*ceil(K/2)` layoutをworkspace sizingとexecuteで共有し、scale offsetの重なりを修正した。host layout回帰／両target syntaxはPASS、両release buildはexit0／source before-after一致。V620公開pair probeはM64,65,127,128,129,512の独立2matmul比較・oracle・repeat・量子化1回・cleanup0をPASSし、旧M1／FP8経路もPASS。full-model同一binary off/on比較へ進む。証拠は`r16b-public-prefill-pair-gfx1030/`。R9700の結果と速度採用判断は未確定。

- r16b R9700公開pairもM64,65,127,128,129,512でoracle／repeat／共有量子化1回／cleanup0をPASS。full-model同一binary off/on単回はV620 prefill39,997.216→40,537.306 ms、decode5,320.554→5,434.482 ms、R9700 prefill19,081.115→19,146.437 ms、decode5,458.362→5,375.957 ms。両GPUともprefill改善を示さず、現時点で既定採用しない。無変更full-modelの再測定で好結果を探さず、実chunk shapeでpair対独立2matmulの同一process費用比較が必要かを判断する。full-modelのtoken／dispatch／memory／cleanup比較とservice復旧記録は集計中。

- r16b V620 full-modelはoff/onで生成／visible token hashとMTP74/107が一致、HIP-only／cleanup0、workspaceとrequest high-waterも一致。total kernel dispatchは83,061→82,837、submission54,985→54,761で重複量子化224回を削減したが、単回E2Eは約1.44%悪化した。証拠は`v620a-r16b-prefill-shared-comparison.json`。R9700既存FP16 serviceも同一設定へ復旧しhealthz／readyz200。
- 次の費用切分けはr16b公開pair harnessをscratchへ複製し、M1024／2048・K5120,N17408でpair対独立2matmulのwarm済み同一process AB／BA各20組を比較する。kernel／selectorを変えず、全出力・oracle／dispatch／workspaceも確認する。単回full-model差を改善へ読み替えるための再試行ではなく、共有による量子化削減とpair実行費用の大小を測る。
- exact fixed-point scratch初回はcompile成功したが、root確認でcontrolが現在選択されるStageK64 lookaheadでなく過去のStageK64 bodyだったため、GPU比較前に差戻した。現行production control／同じlookaheadを基準にし、NaN scale明示ケース・finite scale境界・実M2048を追加して再compileする。初回127→98 VGPRの差を現行既定比の改善と扱わない。

- r16b R9700も同一128token hash／MTP76/104、HIP-only／terminal nonfinite0／cleanup0、workspace・request peak不変。target dispatch78,961→78,737、draft20,173不変。E2E24,574.703→24,565.554 msで単回のほぼ同値。`r16b-r9700-prefill-shared-r1/batch-summary.json`へ集計し、service artifact一致／zero clients／healthz・readyz200／restore_required=falseを確認した。
- 不採用r14 draft-M1／r15 target-M3 Graph実験は、試行・数値結果・CPU増加・採用見送り理由を上記履歴に残して実装から削除する。対象は4 opt-in、draft専用queue／M3 replay bucket、native M3 graph admissionと専用検査だけ。既存non-MTP M1 graph、r16 pair、queue pool、MTP checkpoint・argmaxは保持する。Rustとnativeを分担して削除し、影響するhost／compile検査を行う。

- V620 pair micro初回はM1024／2048で全出力oracle最大0 ULP／repeat／cleanup0、shared/direct時間比0.989804／0.986717。ただしrootのharness確認で、実際はAB10＋BA10組（合計20）であり、direct側だけ各matmul後にwaitしていた。通常pending segmentのgate/up連続enqueueと異なる追加fenceを含むため、この約1%差を量子化共有の採用根拠にしない。旧証拠を保持し、direct2個enqueue後の最後のcompletionで同期／両completion安全解放、AB20＋BA20と順序別raw timingを記録するr2へ修正する。

- Graph実験のnative削除は完了し、M1のみのgraph runtimeは既存HEADと一致する。r16 projection pairとqueue pool／checkpointは保持。public runtime host1/1、両target direct HIP compile、更新したmanifestでH3 31 test＋466 subtestをPASSした。Rust側の候補削除とfocused検査は継続中。
- fixed-point r17を現行lookaheadへ揃え、17,075 byteのcontrol section一致（SHA `3c28200e573acfccc5e52ae5a0ed28ed6d5c5fa65f1d52b95d72467a4d8ae754`）を確認した。candidateは194 VGPR／107 SGPR／LDS30,720 B／SGPR spill53、control228 VGPR／82 SGPR／LDS30,720 B／spill0。resource値だけで高速化／棄却を決めず、実機比較へ進む。独立long-double oracle、M257／2048、両tuple、全254 finite scale encodingとNaN scaleを用意し、N1方針に反する有限出力のcontrol bit一致gateは除去して差の報告へ変更した。

- r17 exact fixed-point候補はR9700 exact gfx1201でexit0、全6caseのrepeat／finiteまたはNaN分類／各64点long-double oracle最大0 ULPをPASSしたが、candidateは現行lookaheadより約1.7〜1.85倍遅く性能棄却した。M2048 wideはAB13.523→23.146 ms、BA12.900→22.145 ms。53 SGPR spillはprivate segment0／scratch命令0でVGPR laneへの退避だが、ISAにはscalar spill/reload474命令があり、整数累積と追加管理の費用削減にはつながらなかった。spill単独を実測差の唯一の原因とは断定しない。
- all-finite scale境界caseでは35,651,584出力のうち975個がcontrolと異なり最大47 BF16 ULP。これはN1候補の数値差として報告し、BF16 tieだけの差とは呼ばない。oracleは64点sampleで最大0 ULPであり、差分全975点の精度を独立に検証したとはしない。NaN caseは両者5,376 nonfiniteで分類一致、canonical NaN payloadの違いはhash差として残す。performance棄却のためfull-modelや追加integer-WMMA派生へ進まず、productionへ導入しない。証拠は`r17-r9700-fixedpoint-r1/probe.log`と`fixedpoint-r17/resource-review.md`。

### 次のdecode scratch: ID94 M3のactivation LDS共有

- r10 R9700 decodeでID94は約0.978秒／9,072 dispatch。現行gfx1201のM3 ISAはloop bodyあたり28 global load／168 perm／96 dot4で、そのうちactivationは12 load／72 perm。8 waveが同じactivation blockを別々にdecodeする。現行resourceは93 VGPR／50 SGPR／LDS1,056 B／spill0。
- pinned llama.cppのdecoded activation共有を参考に、M3・K5120,N17408／K17408,N5120の既存2tupleだけで64個のK16 block分をLDSへstageするscratchを試す。3rowの4 packed wordsとfloat scaleで約3,840 Bを追加する。M2／M4は現行のままとする。
- weight prefetch／decode、laneごとのK順、dot4順、FP32 FMA／shuffle reduction、BF16丸めを維持するN0候補。consume前と上書き前に全256threadの同期が必要で、K5120／17408は最大10／34 barrierとなる。activation decodeの重複削減とbarrier／LDS費用を実測で比較する。
- 本体へ先に導入しない。exact current ID94 controlのsource対応、gfx1201 resource、同一process AB／BA各20組、独立oracle／全出力bit一致／finite／repeat／cleanupを確認する。性能改善を確認した場合だけ公開routingと全体測定へ進む。parameter sweepや未検証の数値契約変更へ拡張しない。

- prefillの16-key block softmaxも数値面を調査したが、QKとexpfを保ってもblock max／exp rescale／signed weighted-Vの丸めが変わり、現時点ではN1の誤差非増加根拠を確立できなかった。llama.cppと既存FP16-KV実装の構造だけでMXFP8のN1成立とはしない。約32 KiBの追加K/V LDSも必要で、現段階では実装・既定採用へ進めない。検討記録は`block-softmax-feasibility-r18.md`。N2を無断採用せず、次の実測候補は上記N0のactivation共有とする。

### 次のprefill scratch: FP64外側累積の費用確認

- exact整数累積r17はscaleの整数decode／乗算／退避が高価だった。別候補として、現行lookaheadが作る同じFP32 termをFP64へexact変換し、K昇順のFP64加算1回へ置き換える案をresource compileから調べる。WMMA／scale load・積／prefetchは現行と同じで、最終はFP32へ丸めて既存tensor scale／BF16 stageへ渡す。
- K<=17408ではK16 term数<=1088。標準的な誤差boundではFP64逐次和のgamma_1088と最終FP32丸めの和を、現行FP32 Kahanの約2u32のboundと比較できる。termのexact性と対象範囲を確認してN1根拠を記録する。FP64はexact sumとは呼ばず、NaN／infinityの扱いも通常の浮動小数点演算として検査する。
- gfx1201のFP64命令費用は未測定。4個のFP32 Kahan更新を置き換えても、FP64 throughputの低さで悪化し得る。まず現在のcontrolと同じlookahead構造でcompile／ISA／resourceを比較し、測る価値があれば同一process AB／BA・独立oracle・finite境界／NaN／repeatで評価する。productionへ先に入れず、整数WMMA派生や未検証のfast-mathへ拡張しない。

- pair micro r2は両者ともdeferred queueで1 fence後にcompletionをfinalizeし、AB20＋BA20と全出力oracle／bit一致／repeat／cleanup0を両GPUでPASS。shared/directの中央値比はV620 M1024=0.986753／M2048=0.989570、R9700=0.983256／0.982152。R9700は両順序とも改善。V620 M2048はABで改善、BAで0.52%悪化する順序差が残り、pool中央値だけで全条件の改善を主張しない。
- R9700 paired差の平均±標準偏差はM1024 -0.267535±0.128928 ms、M2048 -0.507077±0.175807 ms。V620は-0.399875±0.381306／-0.515158±0.825747 ms。全体の速度目標達成ではないが、同期条件を揃えて量子化共有の局所費用削減を確認した。通常設定への採用準備として、未対応provider／強制rollback時に実行前に従来分解へ戻せるかを確認する。未確認のfull-model改善を採用理由にはしない。証拠は`r16b-pair-micro-r2/comparison-combined.json`。R9700 serviceは同一設定へ復旧しhealthz／readyz200、zero clients。
- Graph候補のRust削除も完了。Qwen execution77 PASS／既存1 ignored、prepared execution15 PASS、Qwen graph20 PASS／既存1 ignored、core／benchmark checkとdiff検査をPASS。generic M1 replayとMTP checkpoint、r16 shared pairは維持し、4実験env／M3 bucket／draft graph countersを削除した。

### 共有prefillの既定採用とprepare段階の分解

- 局所費用削減と両GPUの数値／memory不変を根拠に、既存の共有prefill scopeを既定有効へ変更する。`SLLM_QWEN38_NVFP4_PREFILL_SHARED_ACTIVATION=0`／不正値は従来分解へ戻す。全体速度の向上や目標達成は未認定のまま、後続の統合測定で確認する。
- HIPのsemantic supportsはnative providerを見ないため、それだけでは強制baseline等の互換性を保てない。native pair prepareは未対応providerをplan予約／workspace／queue submit前にUNSUPPORTEDで拒否する。このprepare限定statusをtyped Unsupportedへ写し、共有routeはsupports拒否またはprepareのUnsupportedだけで既存2matmulへ分解する。
- その他のprepare失敗、submit後／async失敗では分解へ戻らずrequestを失敗させる。cacheのnode ordinalとgeneric M1 graphの記録を維持する。新しい公開APIや汎用capability frameworkは追加しない。
- focused検査はdefault／0／不正値とrow scope、supports拒否／prepare Unsupportedからの2matmul分解、その他prepare失敗／submit失敗の非fallback。native hostでは両targetのM64 forced baseline拒否がnull plan／追加確保・submitなしであることを確認する。

- FP64 outer r18とID94 activation-LDS r18の両scratchをsource固定し、現行control一致を確認。R9700の同一検証batchで順に数値・速度を測定し、終了時は既存FP16 serviceを同じ設定へ戻す。本体への候補導入と正式full-model測定は結果に依存する。

- r18 FP64外側累積はR9700で全6caseのrepeat／finite・NaN分類／各64点oracle最大0 ULPをPASSしたが、現行lookaheadの約4.8〜4.95倍の時間を要し性能棄却。M2048 AB12.623→60.675 ms。finite境界975差分／最大47 BF16 ULPも記録し、sample外まで独立精度を確認したとはしない。productionへ導入しない。
- ID94 activation-LDSは両tupleの全出力host oracle（52,224／15,360点）最大0 ULP、全出力control一致／repeat／finiteでexit0。ただしprobeのBA測定時にcandidate時間をcontrol配列へ格納するラベル誤りがあり、印字されたpool中央値は無効だった。保存済みの順序別raw配列と凍結sourceの対応に基づいてBAラベルを交換し、GPU再実行なしで再集計した。rawとbinaryは変更せず、`batch-summary-corrected.json`に補正根拠を残した。
- 補正後のM3 K5120,N17408はcontrol0.1127985／candidate0.1124990 ms（約0.27%差）で明確な利益を認定しない。K17408,N5120は0.118519／0.1137185 ms（時間約4.05%削減）、AB／BAとも約4%の改善。このDown tupleだけをgfx1201の既存ID94 wrapperへ既定統合する。gfx1030、M2/M4、wide M3は既存bodyを維持する。新provider ID／公開symbol／実験envは追加しない。
- 上記batch後は同じFP16 serviceを復旧しhealthz／readyz200／zero clientsを確認。証拠は`r18-r9700-candidates-r1/`。native M64 forced baseline拒否の追加host検査も両targetでUNSUPPORTED／null plan／追加確保・submitなし、既存matching ID87/89検査とともに1/1 PASS。

### r19: 既定経路への統合検証

- 共有prefillを既定有効とし、prepare限定Unsupportedからの従来分解を実装した。focused検査はshared activation5件、projection pair3件、HIP bridge24件をPASS。ID94はgfx1201 M3 Downだけactivation LDSを使い、既存LUT／providerを維持する。統合sourceのcompileではgfx1201が93 VGPR／49 SGPR／LDS4,896 B、gfx1030は93 VGPR／51 SGPR／LDS1,056 B、両者spill0。
- Rust整形を適用しfmt／diff検査をPASS。native source manifestを同期し、H3は31件＋466 subtestをPASS。`candidate-gfx1030-r19-default-shared-lds`／`candidate-gfx1201-r19-default-shared-lds`のrelease buildはexit0、各buildのsource before／afterが一致した。これはGPU正しさ・速度の証明とは分ける。
- V620のfresh公開ID94 probeはM2/3/4×wide/downの全6caseで全出力oracle最大0 ULP、repeat／finite／provider ID94／解放をPASS。証拠は`r19-public-smallm-gfx1030/`。
- 通常経路の測定settingsはMTP on／幅2だけを指定し、共有prefillのopt-inと削除済みGraph envを含めない。chunk2,048／state8,320、8,192入力／128出力の単回統合測定を開始した。速度目標の正式1＋3判定とは分け、今回の変更による実dispatch・token・resource・全体時間を確認する。
- V620 r19単回はprefill204.224682／decode23.479335 tok/s、TTFT40,153.482 ms／E2E45,562.585 ms。shared env未設定でもdispatch82,837／submission54,761でr16b明示onと一致し、128token hashとMTP74/107も一致した。HIP-only／nonfinite0／cleanup0を確認し、正式達成とはしない。証拠は`v620a-r19-integration-summary.json`。続いて通常APIの短文／SSE／cancel／recovery／8192/128を実行する。
- V620 r19の通常APIはtext／SSE／cancel／recovery／8192/128がHTTP200、shutdown current／retryable／quarantine0でPASS。長文E2E46.419秒、physical VRAM peak32,739,827,712／34,342,961,152 B、GTT peak15,060,992 B（baseline15,020,032 B）。直後のphysical解放には遅延があるためshutdown時のallocator0をphysical baseline復帰と同一視しない。証拠は`.local-artifacts/phase83/p835-v620a-r19-default-r1/`。この新しい既定sourceについてV620の正式1 warmup＋3 measuredを開始した。
- R9700 r19公開ID94 probeも全6caseの全出力oracle最大0 ULP／repeat／finite／解放をPASS。通常単回はprefill416.295754／decode23.456297 tok/s、出力hashとMTP76/104はr16bと一致、HIP-only／cleanup0。局所短縮からfull-model改善を認定せず、500／25未達のため同じbinaryの正式性能反復は行わない。
- R9700 r19の通常APIも5caseすべてHTTP200／HIP-only／MTP／shutdown残留0でPASS。長文E2E22.404秒、physical VRAM peak32,857,436,160／34,208,743,424 B、GTT peak266,670,080 B（baseline266,579,968 B）。証拠は`r19-public-smallm-gfx1201/`、`r19-r9700-default-ordinary-r1/`、`.local-artifacts/phase83/p835-r9700-r19-default-r1/`。終了後は同一FP16 serviceのbinary／run.sh不変、active/running、healthz／readyz200／zero clientsを確認した。
- V620 r19正式1＋3はprefill中央値199.970802530 tok/s（MAD1.193378953）、decode23.828922062（MAD0.000803833）。prefillは200未満で**未達**。TTFT40,991.903 ms／E2E46,321.446 ms、全4回で128token hash／MTP74/107一致、HIP-only／nonfinite0／cleanup0。終了後のphysical VRAM17,215,488 B／GTT15,020,032 B／GPU busy0を確認した。証拠は`r19-v620-formal-summary.json`。丸めてPASSとせず、同一binaryの未変更再測定で目標到達を狙わない。

### 次のscratch: online softmaxのexp(0)除去

- 現行MXFP8 Q8/W16はK/Vを8 query×6 GQA headで共有済みであり、同じ共有を追加する余地はない。代わりにonline softmaxの2個の指数計算を調べた。finite scoreとfiniteまたは初期値-INFのrunning maximumでは、一方の引数が必ず0となる。現行gfx1201 objectは2個の`v_exp_f32`を保持しており、compilerによる除去は行われていない。
- N0候補として、大小比較で非ゼロ側の差だけを同じ順序で計算し、expfを1回実行してもう一方を1.0Fとする。fmaxf、QK reduction、denominator／V積算、最終丸めは維持する。nonfinite score、+INF／NaN maximumは元の2-exp式を使い、-INF同士などの既存nonfinite挙動を変えない。signed zero、subnormal、最初のkeyも確認する。
- 本体変更前にscratchのexact current control対応と両targetのcompile／resourceを確認し、境界を含む数値式probeとattentionの独立oracle／全出力control一致／repeatを検査する。速度は同じ同期条件・AB20＋BA20で判断する。register／occupancy変化は費用の根拠として記録し、それだけで新しい必須gateにはしない。Q4やblock softmaxへscopeを広げず、実測改善したtargetだけを統合候補とする。調査記録は`r20-one-exp/README.md`。
- ID89のM128/N32化もread-onlyで確認した。現行lookaheadのLDSは30,720→25,600 B、明示accumulator／correctionは64→32 float/threadへ減らせるが、prefetchの4 group／256 scale固定境界を修正する必要があり、実VGPR／occupancy改善は未確定。旧single-buffer N32は既にwide低速／down不安定で棄却済み。現行lookaheadと同じ実験ではないものの、現段階で追加compile／GPUを行わずone-exp候補を優先する。調査記録は`r20-n32-feasibility/memo.md`。
- one-exp scratchは両targetのcompileとcurrent control対応を確認し、レジスタ／LDS／spillは不変。V620の10種類の数値式境界は各更新段階のfinite bit／nonfinite分類・独立host oracleをPASS。attention4caseは全出力bit／repeat／finiteをPASSし、prefix1024/M128・1025/M129の独立oracleは最大1 BF16 ULP。長い2caseは独立oracle未実施であり、全出力control比較と区別する。
- V620のcandidate/control中央値比は順に1.00740／1.00674／1.00381／1.01334で、AB／BAとも全case遅いため性能棄却。本体へ採用せずfull-modelも行わない。証拠は`r20-one-exp/v620a-r1/summary.json`。R9700の数値式10caseもPASSしたが、attention host harnessがgfx1030固定で生成されており、target照合で実行前にexit2となった。これはGPU性能結果ではない。as-run binary／ログを保持し、R9700向けhost定義を付けた別binaryで検証を続ける。
- R9700のhost定義だけを修正した別binary `r20_attention_ab_gfx1201_corrected_r2`（SHA `36ba4e7facc48d823cb29ebb75ee2a6e5a08bf40b953fd3b7ceb5e4788dc05e0`）で4caseの全出力bit／repeat／finiteと2caseの独立oracleをPASS。candidate/control比は1.00549／0.998518／1.00413／1.00417。M129の小差はBAで改善せず、主要caseは両順序で遅いためR9700も性能棄却する。one-expは本体へ導入しない。証拠は`r20-one-exp/r9700-r2/`。

### 次の数値解析: block16 online softmax

- r18で未成立だったblock softmaxのN1根拠を、専門的な誤差解析として再検討する。rootの検討では、同じQKとexpfを使い、block最大値へ既存状態を1回だけrescaleしてkey順に加算する方式は、rescale回数と積算時の丸めを減らせる可能性がある。ただしexp引数の丸め、signed V、underflowを含めた非増加boundはまだ証明していない。
- 調査は1件のbounded specialist taskに限定し、scratch memo `r21-block-softmax-bound.md`へ仮定・導出・成立または未解決点を記録する。旧実装より意図的に緩いbaseline boundを作ってN1とせず、pointwise BF16一致を新しい必須条件にも加えない。調査着手時はsource変更／compile／GPUを保留した。他の改善を止めるgateではなく、実装前の数値上の可能性を調べる作業とする。
- 中間解析ではnormal finiteの非増加boundを確認した。各keyの最終係数について、max差の非負距離の総和は旧／blockとも`M-s_i`へtelescopingし、block側の非ゼロexp因子数は増えない。先頭keyと状態rescaleをFMAでまとめればdenominatorの丸め深さも増えず、以降のV積算を`fmaf(weight,V,acc)`とすることで積の独立丸めを減らせる。signed Vは`Σw_i|V_i|`に対する絶対誤差で比較し、cancelled sumへの相対誤差やpointwise一致とはしない。
- この中間結果に基づき、**scratchだけの実装・compileを許可**した。本体採用のN1分類はunderflow／nonfiniteの扱いを確認してから行う。K/Vを16行、scoreを`score_tile[16][48]`へ保持する35,840 B LDS案を使い、QKの8-product reduction／5 shuffleを維持する。scoreをper-thread配列へ置く案は16 VGPR/threadの増加が予想されるため採らない。layout案の最小同期はK/V投入後と上書き前の2回／tileだが、例外flag共有には追加同期を使う。scopeは既存Q8/W16のgeometryで、全attentionへの一般化やparameter sweepは行わない。
- 実験用controlはr19の現在sourceと対応させ、gfx1030 B4／gfx1201既存1行経路を保持する。candidateはblock先頭のdenominator／V更新を旧式のFMA構造で行い、以降だけkey順に加算する。非有限値では保存済みscoreから従来のkeyごとの更新へ戻す案を解析と照合する。新しいshared layout、tail／mask、全出力oracle／repeat、数値差、resourceと速度を検証し、本体へ先に導入しない。layout記録は`r21-block-layout.md`、rootの導出草案は`r21-block-coefficient-sketch.md`。
- 初回scratchは両targetでcompile成功、LDS35,844 B／spill0。gfx1030はcontrol106→candidate135 VGPR、occupancy9→7 waves/SIMD、gfx1201は99→132 VGPR、12→10 waves/SIMDとなった。速度改善の証拠ではない。この版のguardはnonfinite scoreに限られるため、decoded V／running state／overflowを含むguardを追加してからGPU比較へ進む。初回blockを従来式で処理し、その後は更新前に非有限値と`|old_acc|+key_count*max|V|`の有限・半FLT_MAX以下を確認する案を実装中。初回compile記録は`r21-block-softmax/README.md`。
- guard-r1では初回blockの従来式、非有限K/V・score・state・gap、exp結果の範囲、accumulatorのoverflow余裕を更新前に確認する処理を追加した。失敗時は保存scoreから現block以降を従来式で実行する。root確認で共有flagの同時ordinary-read／atomic-writeを除去し、V最大値のatomic集計を4,096回から16waveの集計へ変更した。source SHA `526cd2b3e8e2f2e43ec7873c4f3980e0569d8c1e112bf75c2d613fd368f6e34c`、両target `-Werror` compile成功、VGPR129／125、occupancy7／10、LDS35,844 B／spill0。GPU未実施で、採用・速度改善は未認定。境界15/16/17・31/32/33、NaN K/V、overflow guardを実際に発火させる有限値fixtureを含む比較harnessを準備している。

### r21 guarded block16の実機結果と次の費用切分け

- guard-r1と固定harnessは両GPUでexit0、4通常caseの全出力finite／repeat、2case全oracle・長い2case各3,072点oracle、active key15/16/17・31/32/33の候補単独oracle／repeat、NaN K/V分類、`2^123`の有限Vでguardを発火させるfixtureを通過した。通常caseの候補／control差は最大1 BF16 ULPであり、bit完全一致は要求していない。full-model比較は実施しない。
- AB／BA各20組でV620は約2.70〜2.84倍、R9700は約3.0〜5.3倍遅く、**両targetで性能棄却**。V620のprefix7168/M1024はraw配列の通常中央値で129.1735→348.2135 ms。harness表示は偶数個の上側中央値を使っていたため、rawから中央2値平均と40点poolを別途計算し、raw／表示値を保持する。再測定で置き換えない。証拠は`r21-v620a-guarded-r1/summary.json`と`r21-r9700-guarded-r1/`。VはVRAM/GTT baselineへ復帰し、Rは元service・candidate identity不変、healthz／readyz200を確認した。
- N1解析は初回blockのuncontracted従来式を共通prefixとして修正し、underflowを含む一様なworst-case boundとpointwise非保証を明記した。guardだけの重複expはstateへ入らないため誤差因子には数えないが、実行費用には含まれる。重複expを省ける範囲契約の追記後の解析memo SHA `720583d7f2f3631c97dafbd49c6de4c082da79d12df5ee02cde48b42892eecd7`。
- 次のr22は**guard内の重複expだけを除く1候補**へ限定する。更新前に各gapが有限・非正であることを確認し、stateを作る既存HIP expfは維持する。非正有限引数に対するexpの範囲契約と既存exp(0)実機証拠によりoverflow boundの前提を保つ。B16／LDS／QK／加算順／fallbackを同時に変更せず、B4等のparameter sweepは行わない。scratch compile・resource比較から開始し、別source／binary identityで検証する。費用調査は`r21-followup-cost.md`。
- r22は両targetでcompile/link成功、source SHA `634698009caac0c4c00430b72f3f2ecf8389a5d829d0b2b25dc2db04bca24ded`。VGPR129／125、occupancy7／10、LDS35,844 B／spill0はr21と同じ。rootはkernel差分が重複expの除去と有限・非正gap確認に限られ、harness差分がsymbolと偶数中央値の修正だけであることを確認し、同じGPU matrixを開始した。未変更r21の再測定ではない。
- r22実機は両targetでexit0、同じ13caseの数値・finite／分類・repeatをPASSした。しかしV620は従来controlの約2.56〜2.70倍、R9700も約2.95〜5.17倍遅く、重複expを除いても差が大きいため両targetで棄却する。V620のprefix7168/M1024は129.4015→331.837 ms（raw40点pool中央値）。本体へ導入せず、このblock16方式の追加調整を止めて行列演算へ戻る。Rの元serviceは同一設定へ復旧しhealthz／readyz200、zero clients、identity不変。証拠は`r22-v620a-no-duplicate-exp-r1/summary.json`と`r22-r9700-no-duplicate-exp-r1/`。

### r23: 現行lookaheadのN32出力タイルを調査

- r19 profileのprefill主要費用ID89について、現在のStageK64・2-parity lookaheadをM128/N64からM128/N32へ狭めるscratchをcompileから調べる。旧single-buffer N32は性能棄却済みであり、その資源・速度を現在のlookaheadへ読み替えない。先行read-only解析は`r20-n32-feasibility/memo.md`。
- 各出力のK16順、WMMA、Kahan、scaleと最終丸めを維持するN0候補。出力列だけを別workgroupへ分け、weight prefetchを4→2 group、weight scale planeを256→128要素へ明示的に境界化する。新device symbolとN32 gridをscratch内で用意する。本体selector・公開APIは変更しない。
- scopeは既存gfx1201の2tuple、parameter sweepなし。compilerのVGPR・LDS・occupancy／spillを現行controlと比べ、資源改善の見込みがなければcompile結果までで止める。見込みがあれば数値・実chunkのAB／BA比較で採否を決め、compileだけで性能改善を認定しない。
- 生成時にcontrol symbolの欠落と置換後の構文ミスがあり、元bodyからの生成と差分確認へ戻して修正した。成功版ではoriginal include全文がbyte一致のprefixとして残り、candidate差分はN32分割とweight staging境界だけ。gfx1201でVGPR228→163、SGPR82→80、compiler occupancy6→9 waves/SIMD、LDS30,720→25,600 B、spill0を確認した。M127/129/257/1024/2048・2tupleとNaN scaleのharnessを準備し、N1実験から継承された比較条件をN0の全finite出力一致・両者repeat・nonfinite分類・独立oracleへ修正した。R9700の全11ケースはexit0、全出力pair一致、両者repeat、finite／nonfinite分類、独立oracle検査をPASSし、sampled oracleの最大差は0 ULPだった。しかしcandidate/controlは全shape・AB/BA両順で1.058〜1.227倍であり、性能棄却した。M1024 wideは約6.75→8.28 ms、downは約6.80→8.30 ms。harnessは20回の上側中央値だけを保存しraw配列がないため、通常の偶数中央値は復元不能という測定上の制限を残す。両順とも大きく遅い棄却判断は変えず、修復目的で未変更再測定は行わない。証拠は`r23-r9700-lookahead-n32-r1/`。serviceは同一binary／設定へ復元し両HTTP endpoint 200、client0とhash不変を確認した。

### r24: exactな4項groupとKahan補正の組合せを調査

- rootとbounded agentの解析で、finite E2M1 K16 dotは`d*2^-2, |d|<=2304`、2つのE4M3FN scaleを含むtermの整数係数は最大518,400（19bit）であることを確認した。4termの共通整数表現における指数差が3以下なら、絶対値和は最大16,588,800で`2^24`未満となり、signedな中間和もFP32でexactに表現できる。全254 finite scale codeの64,516組を整数boundで確認した記録は`r24-root-integer-bound.json`。
- exact groupに対してKahanを1回行うN1候補であり、旧Kahan stateやtokenとの完全一致を条件にしない。`Σ|group|<=Σ|term|`により標準的なKahan誤差boundを増やさない方向を保つ。最初の調査memoにはN0一致との混同があったため訂正し、`correction==0`や既存accumulator加算の全exact性は必須条件にしない。正しい解析は`r24-exact-partial-feasibility.md`。
- r23 N32 bodyをscratchの基準とし、scaleが有限・非zero、row側とcolumn側の指数rangeの和が3以下の場合だけ4termをgroup化する。条件外は既存の4回のKahan更新を保つ。K16 WMMA、scale、tensor scale、最終丸め、入力集合を変更しない。guardを計算前に決め、4term全部ではなく出力ごとのgroup accumulator1個（N32では16値/thread）を保持する。
- 追加registerとguard費用が利益を消す可能性があるため、unique scratch symbolのgfx1201 compile・resource確認までを先に行う。r23自体のGPU性能は別に判断し、この数値解析を速度改善の証拠にはしない。source／GPU本体への導入、parameter sweepはまだ行わない。

### r19 R9700 profileの更新

- 現在の既定binary SHA `c4e6fcdecb81d259d080b1fde3cb4e905eae1ca4ed62917bf7df9f4a8dee9c47`で、MTP幅2／8192/128／chunk2048／state8320のprofileを1回実行した。exit0、HIP-only／出力hash一致／cleanup0。profile付きwall値を速度受入には使わない。終了後は元のFP16 serviceを同一設定へ復旧し、healthz／readyz200／zero clientsを確認した。
- exact-symbol durationの合計ではprefillの主費用はID89 lookahead約9.034秒、Q8/W16 attention約5.321秒、FP8 MT128約0.971秒。decodeはID94約0.928秒、FP8 MT16の2種が約0.504／0.459秒、staged32 attention約0.415秒、FP8 MT64約0.366秒。MTP prefix primingを含む既存の処理境界を照合した相対的な順位として用いる。
- raw timestampには同一Queue4／Stream3で30,044件のstart順序逆転と63,477件の隣接区間重複があった。dispatch IDは102,582件すべて一意。decode同streamのduration合計2.997秒に対して見かけの区間unionは0.884秒となり、絶対的な時刻配置をGPU占有率／同stream並列実行／hostまたはqueue idleの証拠にしない。最初のreportのunion／gap表現はこの制限を明示する形へ訂正し、raw／元analysisは保持した。追加GPU traceやROCr内部調査は行わない。
- 証拠は`r19-r9700-default-profile-r1/profile-report.md`と`trace-placement-diagnostics.json`。raw trace SHA `353ab37606e2a2c9addfbb766d5d58f2efa4aab386d096fa7b6e68d8cb1f2b28`。相対的な主要費用は旧観測と大きく変わらず、block softmaxのscratchを進める判断を維持する。

- r24の初回生成はK16 loopの閉じ位置とcontrol body欠落を修正する必要があり、rootがr23 include全文を保存した新しいr2 sourceへ組み直した。gfx1201 compileは成功し、VGPR180、SGPR107、8 waves/SIMD、LDS25,600 B、spill／scratch0。include SHA-256は`4662aac7963cb80f5811e12a5339ce7cf2319fe14aa1f0eeadbb481cd49c5af9`、kernel objectは`77991e7ef51e7e6ea160474fca98f93c4c8d5205c7d966b30f7690e96bcdcb0b`。N32単独は既に棄却したため、比較controlには現在のN64を用いる。harnessは20回AB/BAのraw配列と通常の偶数中央値を保存し、指数差3／4、E4 subnormal、zero／NaN fallbackを含む11ケースを実行する。host側のbounded入力sampleでgroup適用率も記録し、timing kernelにcounterは追加しない。R9700 GPU結果は下記のとおり性能棄却。

- r24 R9700の11ケースはexit0、全出力repeat／finite・nonfinite分類と各64点の独立oracle検査をPASSし、sampled oracleは両者最大0 ULP。全finite境界のM2048ではcontrolとの差がwide481出力（最大128 BF16 ULP）、down239出力（最大41 ULP）あり、N1の診断値として記録する。これは全差分点でoracle一致を確認したという意味ではなく、full-model品質同等性の証拠でもない。
- 速度は全case・AB/BA両順で約2.4〜2.6倍遅い。group適用率はhost bounded sampleで通常入力が約2.4〜2.5%、全finite境界が約93.8%、指数差3とsubnormal fixtureが100%、指数差4が0%。100%適用できるfixtureでも約2.4倍遅いため、**この実装は性能棄却し追加調整・full-model実行を行わない**。N32単独も棄却済みであり、本体はr19のN64を維持する。
- 証拠は`r24-r9700-exact-group-r1/`、harness binary SHA `b635ac08445652d2f8fa74b42f9a7cd2039fb9160984b375981bc56b3202c1f1`。raw20配列は出力末尾の閉じ括弧欠落・小数3桁というformat上の制限があるが、各20値を保存しており通常の偶数中央値を再計算できる。rawは修正せず、表示丸めの精度を超えた主張はしない。数値解析補足は`r24-root-r2-numerical-review.md`。

- r24測定後は同一service unit／binary／設定へ復旧し、healthz／readyz HTTP200、client0、serviceとcandidateのbefore／after hash不変を確認した。集計の初版はpooled40にも20要素用の中央indexを使っていたため、raw配列を保持したまま要素数に応じた中央2値へ訂正した。通常のAB／BA中央値は変更なく、pooled比も約2.39〜2.63倍で棄却判断は同じ。

### r25: 重複するM64実験を避け、small-Mのquarter scaleを共有

- M128/N64をM64/N64・4wave／128threadへ変更する案は、既存`nvfp4-prefill-m64-lookahead/`と同じgeometryだった。既存実測は主要M1024で約13〜16%遅く、VGPR228→250、2CTA合計のLDSとweight stagingも増える。pinned llama.cppの128thread設定だけを理由に再試行せず、build／GPUは追加しない。調査は`r25-m64-feasibility.md`。
- 別のN0候補として、ID94の`((float(block_sum)*0.25)*activation_scale)`を`float(block_sum)*(0.25*activation_scale)`へ変更するscratchを作成した。後者のquarter scaleをrow/blockごとに一度求めて4出力列へ再利用する。M2/3/4と両tuple、gfx1201 M3 Downの既存activation LDS選択、dot4／FMA／shuffle reduction／最終丸めは維持する。新しいproduction経路・ID・envは追加していない。
- `|block_sum|<=2304`とE4 mantissa最大15により積の整数係数は34,560以下で16bitに収まり、quarter scaleの最小非zero値2^-11もFP32 normal。両括弧順がexactであることを解析し、全254 finite E4 code×整数dot -2304..2304の1,170,686組はhost FP32 bit一致、NaN2codeの9,218組は分類一致を確認した。これはhost式の検査でありGPU／full-model証拠ではない。
- gfx1030／gfx1201ともstrict compileに成功。両者VGPR93、SGPR51／49、spill0を維持した。kernelの全M分岐を含むstatic ISAではquarter literal出現がV62072→18、R970079→19となり、重複乗算削減を確認した。static命令数を実行時間の改善率へ読み替えない。sourceは`r25-smallm-quarter-fold/`、candidate SHA `5bad1e2d89cdef8407df307191237fd3df1e81c94cd409203660f9030a478569`。
- 受入は既存N0候補と同じく、exact targetの全出力finite bit／repeat／nonfinite分類・独立oracle、AB/BA両順での速度比較とする。raw配列は桁を保持し、偶数中央値は要素数に応じた中央2値から求める。GPU結果と本体への採否は未判定。

- r25の両GPU probeはexit0。M2/3/4×2tupleの通常6ケース、all254finite scaleのM3 Down、activation／weight NaNの計9ケースでfull finite pair bit一致／repeat／分類をPASS。通常6ケースとNaNケースの独立long-double oracleは最大1 BF16 ULP。allfiniteのoracleは既存controlのboundを新たに必須化せず診断値として保存し、観測最大は両者1 ULPだった。初版harnessのNaN時finite pair比較漏れ、逐次FP32 oracle、中央値の意味をGPU実行前に修正し、source SHA `3539f24e0a8ab8d70b5d4c42f52fc780c648f22d2632b28d9065ba010b6858db`を固定した。
- V620は全6ケース・AB/BA両順で改善し、pooled40所要時間は約0.80〜1.88%減少。wide M3は0.082881→0.082001 ms、Down M3は0.089441→0.087961 ms。gfx1030の既存ID94だけへquarter scale共有を統合する。R9700のpooled比は0.9953〜1.0034倍で方向が一貫せず、gfx1201の計算は変更しない。full-model改善率へ換算しない。証拠は`r25-v620a-quarter-fold-r1/summary.json`と`r25-r9700-quarter-fold-r1/timing-analysis.json`。R serviceは同じunit／binary／設定へ復元、両HTTP200・client0、hash不変。VはVRAM／GTTがbaselineへ戻った。
- Vのみのsource統合後に両targetを再compileした。gfx1030の実命令は測定済みcandidateと一致し、gfx1201は末尾paddingと配置に伴うPC-relative LUT参照offset以外が従来と一致した（`r25-smallm-quarter-integration/normalized-code-correspondence.json`）。source manifest同期とH3の31件＋466 subtestをPASS。engineのfresh full build、公開GPU経路とfull-model性能はまだ未実施であり、最新full-model証拠はr19のまま。

### r26: Q8/W16のwave共通causal判定

- GQA6／8query tile／16waveでは1waveの3itemが同じquery rowを共有する。row／valid／causal limitをwaveごとに求め、4か所のkeyごとの重複判定を置き換えるN0 scratchを作成した。無効rowの加算をguardし、QK／softmax／Vの演算順・key順を維持する。
- strict compileは両targetで成功。gfx1030 SGPR73→55、VGPR106→102、LDS8,192 B、occupancy9→9。gfx1201 SGPR62→46、VGPR99→95、LDS2,048 B、occupancy12→16。spill0。causal用64bit比較は3→1となり、compilerが既に同じhoistを行っていたわけではない。これは速度実測ではない。証拠は`r26-wave-causal-hoist/README.md`。
- 初版harnessにN1実験由来のfinite pair比較緩和と上側中央値が残っていたため、rootが別名r2へ修正した。全head／全finite出力一致、両者repeat／nonfinite分類、従来の独立oracle範囲を維持し、20 interleaved ABBAとraw9桁・通常の偶数中央値／pooled40を使う。修正source SHA `8115bce25b43cf328438fc9185d9d8fe39732b8963085e132e2e8931c581afb7`。両GPUの7ケース（通常4・NaN K/V・large finite）をexit0で完了した。全finite出力pair／repeat／nonfinite分類、通常2ケースのfull独立oracleと残る5ケースの各3,072点sampled oracleをPASSした。全headのpair差は0 ULP。

- V620は全4通常ケース・両順で改善し、pooled時間約1.44〜2.44%短縮。prefix7168/M1024は129.566→127.698 ms。R9700も全4ケース・両順で改善し、同じ長prefillは75.000→61.362 ms（約18.18%短縮）。短い／M128ケースは約1.2〜1.5%のpooled改善であり、M1024の倍率を全shapeへ一般化しない。両targetの既存Q8/W16 kernelへsource統合し、元symbol／launcher／selectorを維持する。証拠は`r26-v620a-causal-hoist-r1/summary.json`と`r26-r9700-causal-hoist-r1/`。
- V-only r25と両GPU r26を合わせたfresh release buildを`candidate-gfx1030-r26-quarter-causal`／`candidate-gfx1201-r26-quarter-causal`で開始した。両buildはexit0でsource before／after一致、H3は31件＋466 subtestをPASS。通常CLI/API・公開GPU test・full-model性能をfresh binaryで確認する。局所改善だけで8192／128目標達成とはしない。

- r26通常8192／128の単回は、V620 prefill **204.546499852**／decode **23.414946708 tok/s**、TTFT40,088.606 ms／E2E45,512.606 ms。MTP74/107、出力128token hashはr19と一致し、HIP-only／nonfinite0／fallbackなし／cleanup0。R9700はprefill **444.431 tok/s**／decode **23.622 tok/s**、TTFT18,468.360 ms／E2E23,844.787 ms、MTP76/104、同じ128token hashとHIP-only／cleanup0を確認した。R serviceも同じ設定へ復元済み。単回だけで正式目標達成とはせず、V620は公開GPU test後に1 warmup＋3 measuredへ進む。R9700は500／25未達のため同じbinaryの正式性能反復は行わない。
- 単回証拠は`v620a-r26-ordinary-r1/summary.json`と`r26-r9700-ordinary-r1/`。benchmark SHAはV `a39a1473291fc83824f30b071af965c53d55b648dc969fe17ef6b925499b98ef`、R `1de1f9e5308efe4520ccd98d6672011837a017a8a8976082c928c6ee68eda1e0`。fresh archiveへlinkした既存Q8/W16・ID94公開testは両targetでGPU実行exit0。Q8/W16は13ケース・26 oracle比較・13 provider間比較とdispatch metadata、ID94はM2/3/4×2 tupleの6ケース・全出力oracle・repeat・provider94と解放を確認した。Q8公開test自体にはrepeat比較はなく、scratchのrepeat証拠と区別する。R serviceは同じruntime hash・healthz/readyz200・接続0へ復元した。証拠は`r26-public-validation-prep/r26-gfx1030-gpu-summary.md`と`r26-r9700-public-lease-r1/validation-summary.json`。通常API lifecycleの最新候補確認は未完了。

### r27: scaled BF16 ingressの重複確認

- E2M1×E4 scaleは最大180の整数係数でBF16へexact、K16積和は係数最大518,400でFP32へexactと表現できるため、scaleをoperandへ織り込んだBF16 WMMAを検討した。全4,064 finite operand組のhost表現確認はexactだったが、同じ方式はPhase83の`bf16-scaled-wmma-gfx1201-r1/`で既に測定され、約1.384〜1.880倍遅かった。現在のdoublebufferへ移すとLDS49,152 B（現行30,720 B）になり、再試行の根拠はない。新しいbuild／GPUは行わず、`r27-scaled-bf16-feasibility.md`と`r27-scaled-bf16-proof/integer-proof.json`へ調査を記録する。

### r29: WMMA accumulator layout変換の調査

- gfx1201 ID89の`apply_data_layout<row_major>(contribution)`はROCm 7.14の同一register layout分岐でコンパイル時identityになる。exact symbolのISAにも対応するlane shuffleはなく、`v_perm_b32`はFP4 decodeに由来する。native accumulator indexingへの書換えで除去できる実処理がないため、この案は実装・build・GPU測定へ進めない。scale座標とK16/Kahan順序は維持する。調査のみでproduction変更なし。証拠は`r29-wmma-layout-feasibility.md`。

### r26: V620正式性能

- 1 warmup＋3 measured、通常既定MXFP8／固定sampling／MTP幅2、8192入力／128公開出力、chunk2048／state8320でexit0。prefill中央値 **201.497277642 tok/s**（MAD0.179022440）、decode **23.860923858 tok/s**（MAD0.024427613）で、V620の200／20を満たした。TTFT40,716.720 ms、E2E46,047.376 ms、TPOT41.909526 ms。旧r19未達値を変更せず、新sourceの正式結果として記録する。
- 全4回の128token hashは`e38500f139e36819d6ed273c95f17e9384aba6e623106b162b679535efb2d9c9`でr19と一致。MTP採用74/107、priming込み、HIP-only、terminal nonfinite0、fallbackなし、cleanup0。終了後VRAM17,219,584 B／GTT15,020,032 B／GPU busy0。model品質のBF16比同等性をこの一致から認定しない。
- 証拠は`v620a-r26-default-formal-r1/result.json`（SHA `fb00e6419cfd18ddf1b8d9eefa66f7e2292eb9640cde737d95f03189e2d421ef`）、`r26-v620-formal-summary.json`。benchmark SHA `a39a1473291fc83824f30b071af965c53d55b648dc969fe17ef6b925499b98ef`。通常API5caseは`phase83/p835-v620a-r26-default-r1`で確認中。R9700の目標・Phase全体の残作業は変更しない。

- r26 V620通常APIの5ケース（短文・SSE・cancel・recovery・8192入力/128出力）は全HTTP200、MTP実行・HIP-only・fallbackなし、shutdown current bytes0／retryable0／quarantine0でPASS。cancel要求はauditでもcancelled、その後のrecoveryはcompleted。VRAM peak32,739,844,096／total34,342,961,152 B、GTT15,020,032→15,060,992 Bでmodel/KV規模のspillはない。runner直後のsysfsには非同期解放途中の値が残るが、delayed確認ではVRAM17,219,584 B／GTT15,020,032 B／busy0へ戻った。証拠は`phase83/p835-v620a-r26-default-r1/execution.json`と`delayed-post-exit-memory.json`。source・build入力と両target全3binaryのhashがr26 build identityに一致することを再確認した。

### 次のR9700候補の事前調査

- r30: 現行ID89 lookaheadの既存2tupleでK/Nは64整列していても、runtime row／column／K境界判定がISAに残る。Mが128の倍数のfull interiorだけを対象にK/Nを静的化し、tailは現行bodyを維持するscratchを準備する。演算順・K16/Kahan・dtypeを変えず、モデル名による分岐は加えない。速度効果は未確認。証拠は`r30-id89-aligned-feasibility.md`。
- r28: one-wave M16/N64のsmall-M WMMA案は未実装。ID94のprefetch2は同じlane accumulatorへ2項ずつ加えるので、K5120/17408のlane内項数は10/34である。初回メモの5/17を訂正した。有限E2M1 K16 dotとE4 scaleの完全な項は整数係数最大518,400でFP32 exact、現行treeの標準集約boundは約15u/39u、Kahan候補は約2u＋高次項と比較できる。ただし実際のWMMA命令・順序・paddingの検証と速度確認は未実施であり採用済みとはしない。pointwise一致や新しいsigned-zero gateは追加しない。現時点ではr30を先に評価する。証拠は`r28-smallm-wmma-feasibility.md`と修正済み`r28-smallm-wmma-numerical-feasibility.md`。

### r26: V620のMTPなし／あり正式比較

- 同じr26 binary・model・8192/128・MXFP8・T1/P.95/K20・seed123・chunk2048/state8320で、MTP offも1 warmup＋3 measuredを完了した。offはprefill **225.515888298 tok/s**（MAD0.324669247）、decode **14.284310835 tok/s**（MAD0.007139759）、TTFT36,347.086 ms、E2E45,238.011 ms。全4回128公開token、HIP-only／nonfinite0／fallbackなし／cleanup0。
- MTPありはdecode **1.670429倍**だが、prefillのcompanion priming込みではE2E45,238.011→46,047.376 msと約1.79%長い。この8192入力・128出力の全体時間まで改善したとは主張しない。ユーザーのprefill200／decode20はMTPありで達成しており、評価指標を変更しない。
- 最新r26のV620ではMTP off/onの全回128token hashが`e38500f139e36819d6ed273c95f17e9384aba6e623106b162b679535efb2d9c9`で一致した。Phase83旧r26 binary／chunk1024の`phase83/mtp-output-comparison.json`では差があったが、Phase83.5 r26／chunk2048の今回とは別identityである。1fixtureの一致を全入力・BF16品質同等性へ一般化しない。
- 証拠は`v620a-r26-no-mtp-formal-r1/result.json`、`r26-v620-no-mtp-formal-summary.json`、`r26-v620-mtp-formal-comparison.json`。off result SHA `10842daceb945171af871e3c1d3580a28f6efeb9473f6f927cbea9b88d843a50`。R9700の最終同条件比較は追加最適化後に残る。

### r30／r28 scratch compile

- r30 aligned ID89はstrict gfx1201 compile exit0。元genericと同じTUで比較し、両static K/N symbolともVGPR228→212、SGPR82→46、LDS30,720 B維持、occupancy6→7 waves/SIMD、spill0。静的命令数3,098→2,071、条件分岐120→46。GPU速度・数値検証は未実施なので、本体へはまだ統合しない。最初のcompile失敗ログを保持し、修正後identityと区別する。証拠は`r30-id89-aligned/`。
- r28はrootが現行ID89のnon-lookahead StageK32 bodyからwave8→1／launchbounds256→32だけを変更したM16/N64 scratchを作成した。先のfeasibilityで見積もったtwo-parity lookaheadではなくsingle-parityであり、過大なprivate prefetchを避ける一候補として評価する。厳密compile exit0、128 VGPR／51 SGPR／LDS3,200 B／10 waves／spill0。guarded padding・K16 contribution／Kahan順序・epilogueを維持する。M2/3/4に対するGPU oracle・nonfinite分類・性能は未実施。証拠は`r28-onewave-stage32/compile-summary.json`。object SHA `4ae65fd72a414ebe256e97c289d93af30ee0eec0f52fe4ab8303c2a8737494cb`。両案ともproduction未変更。

### r30 GPU結果と採用判断

- exactgfx1201で10ケースexit0。8 alignedケースの全finite出力bit差0／repeat／nonfinite分類PASS、各64点long-double oracle最大0 ULP。M257の2ケースは候補を呼ばずcontrol fallbackを確認した。NaNをM257だけに置いた初期harnessはGPU前に訂正し、aligned M256でも両tupleを確認した。
- 通常M256/M1024のpooled40所要時間比(candidate/control)はwide M256 **0.984355348**、down M256 **0.984049281**、wide M1024 **0.984420813**、down M1024 **0.987965841**。すべてAB/BAの両順で短縮。M2048全finite scaleも0.985861796／0.987773831。9桁raw20配列から通常の中央2値平均とpooled40を再計算した。微改善だが一貫したため該当shapeへ採用する。whole-model500/25達成とは別である。
- GPU probe PID1029742はexit0。root wrapper session90003もexit0、元service同一hash・healthz/readyz200・接続0へ復元済み。初回preflightはrocm-smiのUUID表示形式差でサービス停止前に失敗し、sysfs数値照合へ訂正してから実行した。失敗前の記録は`preflight-display-mismatch/`へ保持。証拠は`r30-r9700-aligned-r1/summary.json`。productionへの統合と公開経路の確認へ進む。

### r28 one-wave StageK32の性能棄却

- exactgfx1201の9fixtureはexit0／RESULT=PASS。通常M2/3/4×2tupleは全出力long-double oracle最大1 BF16 ULP、finite／repeat PASS、候補とID94のbit差0。全254finite scaleのM3 Downは各providerのoracle最大1 ULP、候補間2出力差・最大1 ULPを診断値として記録する。activation NaN／weight NaNは分類・repeat一致（nonfinite17,408／2）。BF16 full-model品質同等性を認定しない。
- 20 ABBA・raw9桁から再計算したpooled40のcandidate/control時間比はwide M2/M3/M4 **10.586／10.459／8.974倍**、Down **29.235／28.599／22.287倍**。全shape・両順で大幅に遅く、不採用。このone-wave M16/N64 StageK32案の追加調整・無変更再測定は行わず、本体には入れない。低いLDS使用量だけで小Mのpadding／逐次K段数の費用を相殺できないことが分かった。
- 証拠は`r28-r9700-onewave-r1/`。binary SHA `aae7638a50f9de9a114ab9c68371361c3a1f4472f9d8e69292322d2c7edf0483`。child PID1039446 exit0、通常serviceは同一hash・healthz/readyz200・接続0へ復元済み。修正前のlane項数計算を性能／数値の根拠に戻さない。

### r30統合build

- 既存lookahead bodyをStaticK／StaticNでparameter化し、同じ本体からgenericと2つのaligned symbolを生成する形で統合した。host launcher・device symbol metadata・M255/256/257両tupleのselector testを揃えた。provider ID／env／C ABIは維持。gfx1030の選択は変更しない。
- fresh build `candidate-gfx1201-r30-aligned`／`candidate-gfx1030-r30-aligned`は両方exit0、source before/after一致。R benchmark SHA `4eb1c389e71b397a956ace46be039c10c41937f46e938191c3da8fcf6c04e5e5`、V `9563dc9a26390d7506eec26a2e2d3d0514d474d8fa2cb4d1e39ab5a54e5715de`。H3は最初30件＋466 subtestを通過し、追加kernel名のsort／件数assertの更新漏れで1件失敗した。対象一覧全体を確認し、171 symbols／68 expected additionsへ揃えた後、その1件のfocused再検証をPASS。新しい検証gateは追加しない。
- fresh R archiveへlinkした公開prefill testはcompile/link成功。公開経路M255/256/257と通常8192/128単回を次に確認する。R30のモデル全体の速度目標達成はまだ未認定。V620の直近正式性能・API証拠はr26であり、source変更とcode対応の確認を分けて記録する。

### r30公開metadataの不具合とr30b修正

- r30公開prefill testはM255通過後、M256のdevice symbol照合でexit1となった。追加した2つのkernel名が公開metadataの64 byte上限を超えて切り詰められ、追加static_assertも誤って128 byteを許していた。execute/wait後のmetadata比較で失敗しており、失敗時の既定値provider0を「GPU未実行」の証拠にはしない。数値失敗を示す結果ではない。
- kernel名を`sllm_nvfp4_gfx1201_wmma128x64_aligned_k5120n17408_v1`と逆tupleの短い名前へ変更し、両方NUL込み53 byte、static_assertを64 byteへ修正した。公開C ABIのbuffer長は変えず、launcher・metadata・公開test・H3 symbol一覧を同期した。演算は変更していない。H3 symbol契約のfocused testとdiff whitespace検査はPASS。
- 失敗証拠は`r30-r9700-integration-r1/final.json`。通常モデル測定は未実行（skip125）、serviceは同一hash・healthz/readyz200・接続0へ復元済み。旧r30 artifactは保持し、修正後の`candidate-gfx1201-r30b-short-symbols`／`candidate-gfx1030-r30b-short-symbols`を別identityでbuildする。
- r30の旧名archiveと測定scratchのgfx1201 `.text`はbyte一致し、generic／aligned2symbolの正規化ISA差分も空。r26との既存ID87／ID89およびV620 ID87の対応もbyte一致した。証拠は`r30-code-correspondence/`。この比較は改名前archiveが対象であり、r30b build成功や公開経路成功を代用しない。

### r30b公開経路の再検証とR31の実装範囲

- r30b両target buildはexit0、source before/after一致。R benchmark SHA `8f4078fe85af36588bbdab316148dcc176642551265ece75200e5f883c616ebe`、V `badac2212f5524cd1a62be5834a36fae05571abd089ba91bb40ffcfbcc6bfb35`。短縮後の2symbolとgeneric ID89は旧r30の対応bodyとbyte一致（`r30b-code-correspondence/`）。
- R公開probeはM255/256/257×両tupleの6caseをexit0で通過。M256は短縮したaligned symbol、M257はgeneric、M255はStageK32を選び、boundary5×5 oracle最大0 ULP／全finite／全repeat／解放を確認した。binary SHA `9f7c13c68e296c978f35dc8d55904ce82d5e8ca8f68175931e3fae46a8ffca23`。
- 通常8192/128単回もexit0、prefill **446.441534088**／decode **23.561384566 tok/s**、TTFT18,390.212 ms／E2E23,780.446 ms。128token hashはR r26と同じ`ded3d447d4d100eaff932a5c70e8be1e55b84af4bfee6649acde91b6dd60c506`、cleanup0。500/25は未達であり、同じ候補の正式反復は行わず次の変更を進める。証拠は`r30b-r9700-ordinary-r1/result.json`。
- R31は既存FP8 GDN qkv/z pairのM>1入力量子化共有を実装する。M1は既にdefault-onで、当初の「opt-in」調査メモを訂正する。BF16のb/aを混ぜず、MTP companion graph・adapter・multimodalの除外、既存rollback selectorを維持する。同じquantizer／member provider／演算・投入順を共有し、M*K+M*4のchecked workspaceをplanが所有する。hipBLASLt scale pointerの寿命とM1 graph guardを維持し、generic pointer cacheは追加しない。
- 既存の受入方針に従いM1/2/3/4/5/65/2048の公開pair対独立matmul、異なるrow scaleと符号、全出力の解析oracle／repeat／入力更新／解放を確認する。M>1のgraph capture対応を成功条件へ追加しない。通常経路の効果を測定するまで採用済みとはしない。R19 quantizer全体105.142 ms／12,220回から対象pairの実際の割合は分からず、whole-model改善率は未予測。

- r30b wrapperもexit0。元R serviceのhealthz/readyz200・接続0、frozen input hash before/after完全一致を確認した。通常runはMTP76/104、全dispatch HIP・fallbackなし、VRAMはservice停止前と復元後で約22.589 GB（差-4 KiB）。証拠は`r30b-r9700-integration-r1/final.json`とbefore/after digest。

- R31のworkspace調査: ordinary FP8はqueue scratch対象の`matmul_lowp_path`に入らず、分解した各planがM*K+M*4を所有する。pair化は2領域を1領域へ減らすので、M2048・48layerの同じcache key集合では10,493,952×48 = 503,709,696 Bの削減となる。request内cacheはdescriptor／buffer view／dynamic token数／node ordinalを含むkeyで保持されるため、異なるtail／viewのplanは別途増える。これはsource上の差分見積もりで、実VRAM peakの測定値ではない。

### R31共有pairのbuildと検証開始

- coreのexact FP8 M>1 predicateと通常／all_rows両経路、HIP Rust descriptor、nativeのchecked M*K+M*4 workspaceとscale offset／quantizer grid／全providerの動的Mを統合した。gfx1030 prefill providerもordinaryと同じlauncherへ渡す。selector／C ABI／M1 graph guardは変更しない。sourceをfreezeし、両target release buildはexit0、開始／終了source hash一致。R benchmark SHA `0dcc82a2a39d632c824a0600e36134bf272c5d84bfbc4ccd5ff7d6c35d462308`、V `1a02ca9864349c947d5076270626f3ea950e6c1933a677ca842fc12eafe0623f`。
- 対象core test1件、HIP Rust check、Rust fmt、両target native strict TU compileをPASS。旧host testのM>1拒否期待を動的M／shape不一致検査へ更新し、host CTest1/1をPASSした。旧testの失敗記録を新testの成功へ上書きしない。証拠は`fp8-gdn-host-r31-r1/`、binary SHA `4ac96cd1da4de77923021ff5eb8a9613cde72a0be6f1776d1400dc8dd842e2cc`。
- 公開GPU testは7つのMと既存NVFP4共有回帰を同じfresh archiveへlinkした。新しいMのdirect provider期待もgfx1030 ID92／71・gfx1201 ID5で明示した。M1は従来のforced-LDS controlを維持し、M>1は既定providerを検査する。gfx1030公開probeとR9700の公開probe→通常単回leaseを開始した。GPU結果・採否はまだ未確定。

### R31公開GPU成功と通常経路の未適用箇所

- 両target公開testはexit0。FP8 M1/2/3/4/5/65/2048の全出力解析oracle最大0 ULP、shared/direct一致、repeat／入力更新／provider／workspace／解放をPASSし、既存NVFP4回帰も通過した。probe SHAはV `bd4027091f505735c544a440eff4c4c699f3559a760feb7defb6f18a1bd37fef`、R `14cfb6e0aac1e1e55e7de73d0c18dfb36c8310d00b9e185389058e6054b201b1`。
- 通常単回はV **204.473891083／23.533032974 tok/s**、R **450.951833536／23.406136593 tok/s**（prefill／decode）、両方128公開token・従来hash維持・HIP-only／fallbackなし／cleanup0。V TTFT40,107.219／E2E45,503.965 ms、R TTFT18,203.060／E2E23,629.067 ms。異なるGPUのrunを同時進行した単回であり、正式median・微小差の因果推定に使わない。R500/25未達。
- R target dispatchはr30bの78,737から78,545へ192回だけ減った。48layer×4 prefill chunkと一致し、52回のMTP target verifyに期待した共有効果が現れていない。native単体M3のPASSだけで通常MTPdecodeへ適用済みとは認定せず、graph容量shapeと動的token_countのrouting条件を調査する。R31はprefillだけの変更として完了扱いにせず、意図したM>1共有を接続する。

- R31未適用の原因を確認した。graph lowering時のoperation descriptorは容量2048のshapeを保持し、`runtime_binding_view`が実行時token_countへ後から切り替える。新FP8 predicateが静的shapeの行数とM3を直接一致比較していたため、prefillだけ共有されdecodeは分解された。R31bでrank／width／dtype／encodingの契約検査へ修正し、実行時行数の検査は既存views／bindに任せる。同じ容量2048のexecutorからM3を実行し実pair submitを確認する回帰を追加する。Mごとに新graphを作ったR31テストの不足として記録する。

### r33 M64/N128 tileの事前調査

- gfx1201 ID89をM64/N128・8waveへ組み替える案は、正確に同じshapeの既存測定は見つからなかった。ただし安全なpaired-wave mappingではLDS30,720 B／wave当たりaccumulator／WMMA数／全行列CTA数／総staged bytesは現行M128/N64と同じで、activationとweightの比率だけが変わる。近いM64/N64とM128/N128は既に遅く、現段階ではcompile／GPU費用をかける根拠が弱いため保留する。未測定のM64/N128を性能棄却済みとは書かない。証拠は`r33-m64n128-feasibility.md`。

- R31bはcore predicateと同じ容量2048のexecutor回帰だけを修正し、M2/M3/M5をLast／Allの両経路で実行してpair submit1回／分解Matmul0回を確認した。focused core1件・fmt・diff検査PASS。r31とのbuild入力差分は`crates/sllm-core/src/qwen_execution.rs`のみ。両target buildはexit0／source before-after一致、R benchmark SHA `174f8a7ca139b0d8e09b968838c45a3054466f19534c59d876606c68bb4ee168`、V `acf9aeec6ab3d0a2bf1da53657508fdb2efe096f98d066d4a8720aef2488fecf`。
- native archiveと公開test sourceのSHAがR31と一致し、前回公開GPU実行exit0との対応を`r31b-native-evidence-reuse.json`へ保存した。native testを無変更再実行せず、R31b通常モデルでdecodeへの到達と速度を測定する。V runとR model-only leaseを開始し、R wrapperでは過去public結果の再利用を新しいGPU PASSと区別する。

- R31b V620通常単回はexit0、prefill **204.126046921**／decode **23.808602987 tok/s**、TTFT40,170.760／E2E45,505.034 ms。MTP74/107・出力128token hashは従来と一致、HIP-only／fallbackなし／cleanup0。target dispatchはR31の82,645→80,053、追加2,592回＝54検証×48layer減り、通常MTP target verifyへの共有経路到達を確認した。単回値を正式medianに代えず、R側の結果と目標は別に扱う。証拠は`v620a-r31b-ordinary-r1/result.json`と`r31b-v620-dispatch-summary.json`。

- R31b R9700通常単回もexit0。prefill **452.263494221**／decode **23.354685602 tok/s**、TTFT18,148.577／E2E23,586.540 ms。MTP76/104、128token hash維持、全dispatch HIP／fallbackなし／cleanup0。targetは78,545→76,049、追加2,496回＝52検証×48layer減り、両GPUで通常decodeへの接続を確認した。prefillとdecodeをまとめて速度改善とは呼ばず、特にR decodeはr30b／r31の単回より低く、微小差の因果は未確定。500/25へはさらに改善が必要。比較は`r31b-ordinary-comparison.json`。
- ordinary telemetryのVRAM peakはR r30b32,765,149,184→R31 32,161,239,040 B、V r26 formal32,641,597,440→R31単回32,035,487,744 Bへ低下した。native共有workspaceのsource見積もりに整合する方向だが、比較protocolが違うVのformal／単回差やこれらの値を最新API peakの代用にしない。

- R31b R wrapperもexit0、元serviceは同じfrozen input hash・healthz/readyz200・接続0へ復元した。Vは終了後VRAM17,215,488 B／GTT15,020,032 B／busy0へ戻った。Phase全体は未完了で、R目標達成、最終候補の通常API／正式比較、CI／commit／pushを残す。

### r32 compact stochastic MTPの調査

- 固定K20 samplerのmask／ID tie order／inclusive top_p=.95適用後のsupportは最大20と確認した（K64／K0へ一般化しない）。現行nativeは候補をworkspaceに一時保持するが公開結果は16 byteの選択recordのみで、Qwenのrow batchも各rowのM1 selectorを順に実行している。privateなcompact distribution出力なら全語彙CPU readbackを避けられる。
- 現行MTPはdraft argmaxとtarget選択の一致prefix方式であり、coreの`verify_stochastic`は未接続。p/q方式を試すにはdraft分布、独立RNGの扱い、residual選択と既存state rollbackへの接続が必要。固定target分布とGPU sampling方針を保つ実装範囲を次に検討し、まだ採用・速度改善・品質同等性を主張しない。llama.cppのpinned MTPも一致prefix方式なので、そのままp/q実装を移植できるという説明はしない。調査のみ、source／GPU変更なし。証拠は`r32-compact-support-feasibility.md`。

### r35 Q8／24-wave attention候補

- gfx1201のQ8/W16は48 logical queryを16 waveで処理し、1 waveに3 query分のQとV累積を保持する。query tileを8のままwave数だけ24（768thread）にし、1 waveを2 queryへ減らすscratchを作成する。以前棄却したQ12/W24は1 waveあたり3 queryのまま共有query数を増やした別案であり、その結果を今回の測定結果とはしない。
- kGqaRatio=6が2で割り切れるため各waveの2 queryは同じrowに属し、r26のwave共通causal条件を維持できる。lane内productsのpair tree、wave reduction、key昇順、owner-lane online softmax、BF16 epilogue、MXFP8形式は維持する。K/Vは現行gfx1201の1行stagingのまま。register量低下と実行wave数の増加が同期費用を上回るかを調べる。
- 現行body対scratchのstrict compile／resource、M128/129境界・長prefix・実6144prefix/2048query・非finite分類・full pair／repeatと独立oracleを既存harnessで比較する。性能は200ms warmupと20 AB／BAのrawから中央値を算出する。現時点で本体変更・GPU測定・採用なし。sourceは`r35-q8w24/`、harnessは`r35-q8w24-harness/`へ分離する。

- strict gfx1201 compileはcontrol／candidateとも成功し、VGPR95→69、SGPR46→42、LDS2,048 B／spill0。compiler報告occupancyは双方16 waves/SIMDであり、register削減をoccupancy向上とは記載しない。最初のscratchにはbody768threadに対しlauncher512threadが残り、GPU実行前の差分確認で768へ訂正した。修正済みnamespaced objectと8case harnessで数値／性能を比較する。

### r32 private GPU p/qの実装試作

- 現行targetの固定K20・inclusive top_p後の分布pを保ち、draftを分布qからsampleし、採用確率min(1,p/q)、棄却時max(p-q,0)、全採用時target bonusをGPUで処理する。llama.cppのspeculative exampleを式の参照とし、pinned MTP自体からそのまま移植したとは扱わない。通常selectorの16 byte ABI／非MTP乱数列は維持する。
- scratch support recordはversion／status／count／reservedと20 ID・20 f64確率の256 Bとする。最初の248 B調査案にversion領域を追加した。draft／accept／residual／bonusの乱数を分離し、CPUの`verify_stochastic`はoracleとしてのみ使用する。target marginalを維持する検証方式の変更であり、N1の演算誤差低減を理由にsampling変更を承認したとは説明しない。
- request-owned compact bufferの寿命、target／companion queueの完了順序、既存KV／GDN／MTP commit/rollbackへの接続を確認する。queueが違うことだけで新たな同期APIを必須とせず、現行readbackのcompletionが既に順序を保証するかを調べる。まずtiny p/q kernelをscratchで実装・compileし、分布／RNG／不正入力を独立oracleで確認する。現時点で通常生成への統合・採用率改善・品質同等性は未確認。調査は`r32-device-pq-design.md`、試作は`r32-pq-kernel/`。

### r34／r36の追加候補

- r34はID89内でslotに依存しないweight scaleのLDS読込みを8-slot loop外へ移すscratch。演算順／dtypeを保つがcompilerが既に共通化している可能性があり、先にISA／resource差を確認する。既試行のinactive-LDS直接stagingやtile geometry変更は再試行しない。
- r36はgfx1201 Q8/W16のK/Vをwave内registerへ直接loadし、1 keyあたり2回のblock barrierとLDSを除くscratch。query mapping／key昇順／QK reduction／softmax／BF16丸めは保つ。wave間の共有を失いK/V loadとdecodeが16倍になるため、cacheと同期削減の利害を実測する。R35のwave数変更とは別案で、まず現行W16を基準にする。gfx1030のB4 stagingは維持。既存のB4／block-softmax棄却をこの未測定案の結果と混同しない。`r36-wave-local-kv/`へsource生成とstrict compileを保存し、本体にはまだ導入しない。

- r34はstrict compile成功だが、generic／alignedともLDS/global load命令数は減らなかった。alignedはSGPR46→48、命令数2,044→2,055となり、期待した読込み削減が生成されないため統合せず、GPU測定も行わない。`r34-id89-staging/compile-isa-comparison.md`へ保存した。
- R35の8caseは独立oracle最大1 BF16 ULP／全finite pair差0／repeat／NaN分類をPASSした。M128は両prefixで約21.6%短縮したが、M129は約34.3%、M1024/2048は約18.8/19.5%遅い。実prefill M2048には採用しない。M128の限定的な改善を全shapeの棄却／改善に読み替えない。
- R36も8caseの同じ数値検査をPASS。VGPR95／occupancy16／spill0を維持しLDS2,048→0 B、SGPR46→49。M2048/prefix6144はAB／BA両方で約1.6%短縮、M1024は差が一貫せず、M129はpooled約0.9%悪化。M128は約11.2〜13.6%短縮。rootがraw20 AB＋20 BAから中央値を再計算して照合した。`r35-r36-raw-timing-crosscheck.json`。両runのserviceは同一hash／healthz・readyz200／接続0へ復元済み。
- R36追加harnessにrootがprefix0を入れたが、現行Q8 providerのstart_position>=1024契約によりlaunch前にinvalid argumentとなった。kernel correctness失敗とは扱わず、guardを本体で緩和しない。最初のprefillは従来Q4 providerであり、既存scope内の残りprefix2048/4096を確認する。失敗ログは`r36-r9700-prefixes-r1/`へ保存する。R36初回harnessのfooterに旧名R35が残った点も記録し、実source／binary SHAで識別する。

### r32 workspace captureの実装開始

- side-output用private selector APIと6番目のbuffer accounting追加は採用せず、固定K20 final kernelで既存workspace先頭へ256 Bを書き戻す。全候補のLDS読込み／sort後なので入力領域とaliasしてもよく、最小workspace320 Bへ収まる。U8 workspaceはbyte整列しか保証されないためbyte-safe storeを使う。公開16 B結果とK0/K64経路を維持し、coreはprivate capability version確認後に既存D2D copyでrequest-owned slotへ保持する。
- V620 scratch初回のoffset1 alias caseで未使用slot19に古いworkspace値が残る不具合を検出した。全20slotを読込み完了後にzero初期化する修正を加え、両target再compile、V620 r2の8case（vocab1023/1024/1025/2047/2048/2049、mask／additive／tie、非finite／全mask）をPASS。公開16 B control比較は全case bit一致、support ID／確率の報告誤差0。r1失敗は保持した。これはcapture単体の証拠であり、p/q採用率・full-model品質・通常decodeの速度は未確認。
- `token_selector_kernel.hip.cpp`とprivate support header／focused testへの本体統合を開始する。p/q kernel、coreのslot寿命、frontend state接続は引き続き実装中。詳細は`r32-integration-contract.md`と`r32-support-export/`。

### r37 per-key FMA候補

- Q8/W16のdenominatorを`fmaf(d,r,c)`、value累積を`fmaf(a,r,c*v)`へ変えるscratch。QK、最大値、exp、key順、staging、除算／BF16丸めは変えず、accumulator rescale積の独立丸めを除く。denominatorとsigned valueの標準集約絶対誤差boundは非増加となるN1候補で、pointwise改善や全token一致の保証ではない。棄却済みblock-softmaxのlayout変更とは別案。
- strict gfx1201 compile/link成功、VGPR95／SGPR46／LDS2,048 B／occupancy16／spill0。従来の独立oracle・repeat・finite/nonfinite分類を保ち、candidate/controlのfinite bit差は報告だけとする。小さいrecurrence probeでsubnormal／signed zero／極値も確認する。数値分類とGPU性能はまだ未確定、本体未導入。`r37-online-fma-analysis.md`、`r37-online-fma/`、`r37-recurrence-probe/`へ保存する。


### r32／r36統合とr37の結論

- R36追加診断はscratch wrapperのprefix guardを外して実行された。prefix0の結果は現行providerの採用根拠にしない。eligibleなprefix2048/4096はcontrol/candidateのdevice ISAが元scratchとbyte一致し、それぞれ約4.3%／2.4%短縮したため、既存prefix6144の約1.6%短縮と合わせてkernel単体の証拠として利用する。公開wrapper通過の証拠とは分離する。無変更の再測定は行わない。
- R36本体は共通templateのWaveLocalKvをgfx1201・query_count2048のみ有効にし、start_position>=1024と既存head／encoding契約を維持した。gfx1030と他Mは従来経路。両target strict compile成功、gfx1201のVGPR95／spill0を維持しLDS2048→0 B。統合bodyはscratchとresource／関数長が同じでもISA byte一致ではないため、統合後GPU確認を残す。証拠は`r36-integrated-compile-r2/`。
- R37はattention8caseの独立oracle／repeat／分類検査をPASSしたが、全計測shapeで約1〜2%遅いため本体へ採用しない。別の小さいrecurrence probeはlibamdhip64.so.7のloader error（exit127）で未実行。wrapper全体exit0をそのGPU PASSへ読み替えず、棄却済み案の追加検査のためだけに再実行しない。`r37-r9700-online-fma-r1/`。
- R32 captureは既存finalとsupport finalを同一template bodyへまとめて本体へ統合した。byte-safe storeと未使用20slotのzero化を維持し、公開16 B結果を保つ。両target compileとV620本体focused6caseをPASS。private capability versionとheader依存、OFFのfocused CMake targetを登録した。fresh gfx1030 configureでは実HIP compile/link command、expected target、CTest登録を確認した（configureをGPU PASSとはしない）。`r32-support-cmake/configure-evidence.txt`。
- R32のcore／HIP cargo checkは成功。H3初回は並行R36編集によるmanifest source hash不一致で14件失敗したため、source凍結後にmanifestを同期し、同じH3対象の31件・469 subtestをPASSした。Rust fmt、diff whitespace、変更した3文書のlocal link検査もPASS。証拠は`r32-h3-contracts-r2.log`。p/qのGPU検証と通常MTPへの接続は未完了で、Phase完了・速度目標達成は宣言しない。

- R32 sparse p/q scratchはV620A/gfx1030の実GPUで独立dense long-double oracleに対する466検査をPASS（exit0）。width1/2/3/8、採用／棄却／bonus、不正recordとRNG境界を対象とする。binary SHA `ff9277957c066b5e7750407353fe6ecfd9bf729cd1ee7f237d8ee1bccddc885b`、before/after artifact hash一致、VRAM17,215,488 B／GTT15,020,032 B／busy0へ復帰した。既存R serverのV側inert contextはVRAM28 KiB／GTT2,088 KiB・activity0で、合計使用量をVRAM上限と誤認しない。証拠は`r32-pq-kernel/build/gpu-v620a-run.log`とbefore/after manifest。R9700 GPUおよび通常生成への接続・速度は未確認。検証済みscratchからprivate native wrapperの実装へ進む。


### r36統合後GPU確認／r32通常経路の接続

- R36の実production objectを使うgfx1201 harnessはM2048/prefix2048・4096・6144のsampled oracle最大0 ULP、M129/prefix2048のfull oracle最大1 BF16 ULPでPASS。repeat／分類検査、probe exit0、source/object/probeおよび元serviceのbefore/after hash一致、復元後healthz/readyz200・接続0を確認した。両slotが同じproduction launcherを呼ぶため、旧kernelとの比較や速度改善率はこのrunから主張しない。`r36-integrated-r9700-r3/`。
- R36 runner初回はset-u下の未定義LD_LIBRARY_PATH、2回目はharnessに残ったgfx1030期待値によりdispatch前に失敗した。初回metadata heredocの起動ミスも修正した。r1/r2は失敗として保持し、GPU correctness失敗／PASSへ読み替えない。
- R32のprivate native p/q wrapperとRust HIP bridgeを実装し、CMake／Cargo依存とH3のsource・kernel一覧へ登録した。既存公開C headerは変更せず、256 B supportと144 B decisionをrequest-owned bufferで受け渡す。coreのslot保持と通常MTPへの接続を実装中であり、この段階を実行成功とは扱わない。
- frontendでは固定T1/P.95/K20かつ対応する同一sessionの経路にp/qを接続し、Q draftに独立seed domainとchecked position*9+row counterを使う。通常target selectorは従来counterを維持し、queued inputは実際のemitted tokenへ合わせる。RNG変更helperのcore checkは成功、統合後のfocused test／GPU／モデル測定は未完了。

- R32 native reviewで、support／decision bufferの別queue上のactive submissionを拒否する検査を加えた。kernelは全経路でdecisionを初期化するため、重複hipMemsetAsyncを除去した。private wrapperはregistry／accounting lockをfenceまで保持し、投入後失敗ではpoisonして解放を拒否する。両target strict compile、host stub compile、fresh public runtime host targetのbuild/linkとCTest1/1をPASS。host検査をp/q GPU数値証拠とはしない。`r32-pq-host-integration/`。
- H3登録の初回は新kernelを含むmatrix build commandとCargo rerun静的検査の登録形式が未更新で失敗した。実際の新sourceをcanonical commandへ反映し、既存検査が読める明示path bindingへ揃え、31件・481 subtestをPASSした。検査を緩めて通したものではない。`r32-pq-h3-contracts-r2.log`。
- core／frontend／HIP libraryとbenchmarkのcargo check、frontend MTP focused6件（新しいQ RNG domain／counter境界を含む）をPASS。capture形式のcapabilityとp/q verifierのcapabilityを分離し、captureしか実装していないbackendがstate変更後にUnsupportedとなることを避ける。新native private wrapperを直接使うGPU testを追加中。通常測定のreportへ`fixed_k20_pq_blocks`を追加し、実経路到達を確認できるようにした。最終source凍結後の統合検証と通常GPUモデル測定は残る。

- R32 final coreは関連3crate check、decision parser2件をPASS。frontend focused6件もPASS。native wrapperを直接使うGPU test sourceはstrict C++17 syntax／gfx1030・gfx1201 HIP compileを通過し、sourceを凍結した。
- 最初の両target full buildは、native wrapperのuint32 row数とuint64上限の比較が実CMake flagsで常にfalseと診断され、Werrorで失敗した。既存width1〜8検査により最大9×256 Bと証明済みの不要な比較を除去した。standalone strict compileをfull build成功の代用にしない。`candidate-gfx1030-r32-pq-r36/`と`candidate-gfx1201-r32-pq-r36/`のexit101は保持する。
- direct rustfmtによる無関係なcore moduleの整形がaggregate fmt検査で見つかった。両buildが終了した後、canonical `cargo fmt --all`でproject styleへ戻し、無関係なファイルがdirty一覧から消えたこととfmt/diff検査PASSを確認した。修正後は`candidate-gfx1030-r32-pq-r36-r2`／`candidate-gfx1201-r32-pq-r36-r2`を新しいsource identityでbuild中。まだ通常GPU model成功・速度改善を主張しない。

- R32/R36 r2の両target full release buildはexit0、全source before/after hash一致。benchmark SHAはV `c049eb5b543607a04439a5c8d4c84986398d8492612cb02489b62c39bdea29b6`、R `686f46aeeef6282aefbf4a557c89cc4ffa2195faaa07c3d5418cca927f8fadf5`。CLI/serverも同じbuild identityに保存した。新archiveへprivate wrapperのGPU testをリンクし、V620A／R9700の統合検証へ進む。Rは数値検証PASSの場合だけ同じservice lease内で通常8192/128単回を実行する。build成功を通常生成の成功へ読み替えない。


### R32統合GPU検証と通常生成／R33棄却

- R32 private wrapperの統合GPU testはgfx1030／gfx1201ともexit0、正常10・不正record15・wrapper引数10の35caseを独立dense long-double oracleと照合してPASS。width1/2/3/8、144 B decision、cleanupを確認した。V初回はtestが未対応の明示allocation alignmentを要求しkernel実行前にexit2となった。testだけを既定allocationへ修正し、archiveとmodel binaryが不変である対応を`r32-pq-test-only-reuse.json`へ記録した。失敗をGPU PASSにしない。成功は`pq-public-gfx1030-r32-r2/`と`pq-public-gfx1201-r32-r2/`。
- 同じR32/R36 release binaryの通常8192/128単回はV620 **205.765631265／24.125407027 tok/s**、TTFT39,850.946／E2E45,115.187 ms。p/q53/53 block、draft74/106採用、128出力hash `316dc877624f4f03edafe20ff3002ef53dd4f4d90742441bcf23dd61e9ba27ff`。R9700は **448.206726161／24.359533777 tok/s**、TTFT18,315.646／E2E23,529.266 ms、p/q50/50 block、draft78/100採用、128出力hash `7d939adfc7c9b60bb30677bd1c56d462cc6a3b1b86c7512c29913f5d0bb99155`。両方HIP-only／fallbackなし／cleanup0。R serviceは元のhash・healthz/readyz200・接続0へ復元、VはVRAM17,215,488 B／GTT15,020,032 B／busy0へ復帰した。単回を正式中央値へ昇格させず、R500/25は未達。証拠は`r32-pq-v620a-ordinary-r1/`、`r32-pq-r9700-ordinary-r1/`、`r32-pq-r9700-integration-r1/final.json`。
- 旧R31b MTPとの同位置一致はV7/128、R2/128で、どちらも共通prefixは1token。p/qの採否・residual・bonusとdraft乱数を変更した新旧実装比較であり、同一binaryのMTP off/on比較ではない。生成列の一致率を意味的品質やBF16比精度劣化へ読み替えない。Vの同一R32 binaryによるMTP off単回を別途開始した。
- R33 M64/N128・8wave・StageK64のscratchはgfx1201のM256/2048でfull pair差0・repeat／非finite分類PASS、各64sample oracle最大0 ULP。M255/257はcontrol fallback確認のみ。20 AB＋20 BAの中央値はM2048 wide **13.247217→16.147730 ms（21.895%悪化）**、down **13.344299→15.704294 ms（17.685%悪化）**、M256 wide **18.439%**・down **9.164%悪化**。compileでspill0／LDS不変でも高速化しなかったため不採用とする。service復元済み。`r33-m64n128-r9700-r1/`。以前の保留はこの実測棄却で更新する。
- 次のR38はM256/N32・8wave・wave当たり2行tile×2列tile・StageK32をscratchで調べる。既存N32候補は1行tile／waveでM128/CTAだったため同一案ではない。weight再読込みを半減する一方activation再読込みとstage数が増えるため、利益は未確定。R39はp/q時に結果を使わないtarget selectorの16 B CPU readbackだけを省く。companionのreadbackとsupport D2D完了待ちは維持する。いずれも現時点で速度改善・採用の証拠はない。

- R32同一binaryのV620 MTP off単回はexit0、232.421961436／14.307694879 tok/s、TTFT35,279.961／E2E44,156.380 ms、128token hashは従来の`e38500f139e36819d6ed273c95f17e9384aba6e623106b162b679535efb2d9c9`、cleanup0。同一binary／model／fixture／seed／chunk2048でonと比較し、同位置7/128・共通prefix1を確認した。従来のV620 r26でのoff/on一致は旧実装の事実として保持し、最新p/q実装の一致とはしない。decodeは14.308→24.125だがpriming込みE2Eは44.156→45.115秒で、この単回ではMTP有りが約0.959秒長い。正式性能比較やBF16品質同等性ではない。証拠は`r32-v620-mtp-off-on-comparison.json`。

- R32 test修正後のH3契約検査を再確認し31件・481 subtest PASS（`r32-pq-h3-contracts-r3.log`）。canonical Markdown local link検査もPASS。検査範囲はhost契約／文書であり、新しいGPU性能証拠にはしない。

- R32 V620通常APIは短文／SSE／cancel／recovery／8192入力128出力の5caseをHTTP200で完了、HIP-only／fallbackなし／shutdown current bytes0・retryable0・quarantine0。長入力はdraft74/106採用、生成text SHA `85fef6694c7837ecf682f907e80686ce218c48bf3cc4da94f2a899b4965aa1ea`がbenchmarkと一致した。VRAM peak32,135,860,224／total34,342,961,152 B、GTT15,020,032→15,060,992 B、delayed確認でVRAM17,215,488／GTT15,020,032 B・busy0へ復帰。`phase83/p835-v620a-r32-pq-r1/`。cancel auditはcancelledで復旧要求もcompletedだがdraft提案0であり、p/q実行中のキャンセルまで確認したとはしない。次候補の既存API検査では切断を最初のcontentから3回目へ移し、MTP進行後のキャンセルを観測する。


### R39 target側の不要なselector readback削除

- captureへCompanionDraft／TargetVerifyの役割を加え、p/q target blockだけ公開16 B recordのD2H／decodeを省く。selectorのterminal完了と256 B support D2D完了、companionのtoken選択、ordinary／legacy selectorは維持する。target blockは架空のtokenを作らずselectionsなしを返し、frontendは検証済みp/q decisionから出力する。
- suppressionは未完了のtarget captureだけに限定し、成功したp/q検証後はtarget／companion両方のcaptureを消去する。capture不在でのsuppressionは要求をcancelして失敗する。通常samplingへ状態が漏れないことと既存MTP selector回帰をfocused host testで確認した。core／frontend check、frontend MTP6test、fmtはPASS。GPU数値型／kernel／乱数／state commit式は変更していない。
- rootはactive／completed capture境界、出力構築、lifecycle guard、frontend分岐を確認した。`candidate-gfx1030-r39-support-only-r1`／`candidate-gfx1201-r39-support-only-r1`のfull release buildを開始。無変更native evidenceはidentityを照合して再利用し、通常モデルで出力不変・実経路・速度を確認する。まだ採用による速度改善とはしない。

- R39 r1は両target full build exit0、source before/after一致。R32との差はcore／frontendと既に検証済みのtest alignment修正だけで、native archive・test source・build environment不変を`r39-native-evidence-reuse.json`へ保存した。
- V620通常r1は最初のMTP decodeで`output prefix rows 1 are outside 1..=0`となりexit1。private p/q用に公開token recordを省いた後も、既存`slice_qwen_output_rows`が`token_ids.len()`を行数の正本としていたことが原因。hostのrole helper検査だけではstate commitまで覆えていなかった。cleanup current bytes0／poisoned=false／retryable0／quarantine0を確認し、性能PASSとはしない。`r39-v620a-ordinary-r1/`。実block行数に基づくsliceと部分／全採用の回帰へ修正し、r2で再検証する。R r1 binaryの通常実行は行わない。

- R39 sliceを修正し、`pending.token_ids.len()`で得る検証blockの実入力行数を渡す。公開recordなしでもhidden/logitsをこの行数で切り出し、token vectorが存在する場合は実行数との一致を検査する。support-onlyの部分／全採用slice回帰と既存MTP partial計6testをPASS。rootは実commit呼出しでの行数由来と検査を確認し、fmtもPASSした。両targetのfresh r2 buildを開始した。


### R40 MTP state-onlyのKV追加後の不要計算

- graph依存関係を確認した。MTP companionは単一full-attention層で、prefix primingと全採用後の状態合わせはKV状態だけを必要とする。現行state-onlyはterminal projection／Argmaxのみを飛ばし、KV追加後のattention／O projection／residual／MLP／final normを実行していた。後続のMTP層やstateへの依存はない。
- explicit state-only呼出しだけを、成功したFullKvAppendのstate publicationで終了する案を実装する。通常targetのchunked prefillやMTP argmax／device-selectorは全graphを維持する。Q/K/V matmulとcombined AttentionPreprocessはK生成に必要な現行演算として維持し、Q-only削除や形式変更は行わない。append-attention chainの未消費ownerと空segmentの二重完了を避け、既存の失敗・cleanup契約を守る。
- 古いR19 traceではcompanion prefixの16,615 dispatch・合計kernel時間463.158 ms（attention約355 ms）を抽出した。ただし既知のtimestamp順序異常があり、これをwall時間や削減可能なhost待ち時間の上限にはしない。実測するまで目標到達・速度改善率は未確定。既存のtile／MTP Graph棄却案とは異なる依存関係上の不要計算削除で、llama.cppからの直接copyはない。`r40-prefill-work-breakdown.md`。
- R41のsupport workspaceを256 Bずつずらす案はread-only調査まで。targetとcompanionのqueue／完了契約を含む本体検証はなく、R40と同時採用しない。現在のsupport D2D copyは維持する。

- R39 r2両target full buildはexit0／source before-after一致。r1からの差はcoreのslice修正のみ、native archiveとbuild environment不変を`r39-r2-native-evidence-reuse.json`へ保存した。V benchmark SHA `2874b19c1879aad5fb17bcc025166c9ffe1bf2bc7280ed39688e0eadba90766d`、R `19828bba9ba133d0fc6609dd9fac7e01a46815035ff399e2dd4ac3e5b324b0fa`。
- V620通常r2はexit0、**200.494863836／24.508006493 tok/s**、TTFT40,897.757／E2E46,079.798 ms。出力128token・p/q53/53 block・draft74/106採用・dispatch数はR32と一致、HIP-only／fallbackなし／cleanup0。不要readbackを省いて出力を保ったことは確認したが、R32単回とのprefill差（205.766→200.495）もあるため、微小な速度差を正式改善率とは扱わない。`r39-v620-output-comparison.json`。R39r2の通常APIは3回目のcontentで切断する既存smokeにより確認中、R9700通常測定も進行中。
- R38の修正済みharnessはgfx1201全11caseでfull pair差0、repeat／非finite分類、各64sample独立oracleをPASS。実整列ID89を比較相手としたM2048 wideはAB中央値13.565→16.606 ms、down13.498→19.816 msで悪化、M256/1024も遅いため不採用。M255/257は通常StageK32の境界比較で、整列ID89性能証拠へ混ぜない。binary footerの古いgeneric control表記は実際のlauncherと異なるため`r38-control-selection.md`へ訂正を記録した。serviceは同一hash／UUID、healthz/readyz200・接続0に復元。
- 同一R32 binaryのR9700 MTP offは`r32-pq-r9700-no-mtp-r3/`でexit0、**472.424241440／15.746302511 tok/s**、TTFT17,369.468／E2E25,434.900 ms、HIP-only／fallbackなし／cleanup0。offの128token hashは従来の`ded3d447d4d100eaff932a5c70e8be1e55b84af4bfee6649acde91b6dd60c506`、onとは同位置2/128・共通prefix1。同一model／fixture／samplingを照合し`r32-r9700-mtp-off-on-comparison.json`へ保存した。前2回のwrapperはdispatch前の起動失敗として保持し、GPU/model結果に数えない。

- R39 r2 R9700通常はexit0、**447.941875846／23.928987052 tok/s**、TTFT18,324.401／E2E23,631.848 ms。128token hashとp/q50block・draft78/100採用はR32と一致し、HIP-only／fallbackなし／cleanup0。R32の24.360よりdecode単回値が低く、R39の全体速度改善は確認できていない。R500/25未達。serviceは同一frozen hash／UUID、healthz/readyz200・接続0へ復元。`r39-r9700-output-comparison.json`、`r39-r9700-integration-r2/`。
- R39 r2 V通常APIの5caseはPASS。3回目のcontentで切断した要求はdraft提案2／採用2の後にcancelledとなり、次のrecoveryはcompleted。8192/128の生成text SHAはR32と同じ`85fef6694c7837ecf682f907e80686ce218c48bf3cc4da94f2a899b4965aa1ea`、HIP-only／fallbackなし／shutdown current bytes0・retryable0・quarantine0。VRAM peak32,135,852,032 B、GTT15,020,032→15,060,992 B、delayedでVRAM17,215,488／GTT15,020,032 B・busy0に復帰。`phase83/p835-v620a-r39-pq-r2/`。R32のMTP進行前cancelと今回の進行後cancelを区別する。
- R40 explicit state-only KV cutをcoreへ実装し、通常target／MTP draftは既存tailを維持、state-onlyだけ成功したKV publication後の処理を省いた。誤用検査はupload／lifecycle変更前に行い、pending append／空segmentの完了を検査する。focused M1・容量境界・次の通常draftの全tailとKV publication検査、および関連MTP29testをPASS（既存3 ignored）。host fixtureは容量1固定のためM2/M3は実行していない。この範囲を偽って拡大せず、実モデルのfull batch／非整列tailを次のGPU検証で確認する。両target fresh release buildを開始した。

- R40の両target release buildはexit0／source before-after一致。R39 r2との差はcoreのみ、native archiveとbuild environment不変を`r40-native-evidence-reuse.json`へ保存した。benchmark SHAはV `a635631954b725e6525b6b727111a55e8c19994e68e012bac416b169056e587a`、R `437dfb3a0d729d3f24146964fce1f98b979131a4e2702437b7b9ea8bd08928dd`。通常8192/128でfull batchと非整列tail、次draft、採否／出力／dispatch削減を測定中。

- R40通常は両target exit0、128token hash・MTP採否をR39と維持した。V620は **219.257083582／25.143018523 tok/s**、TTFT37,403.147／E2E42,454.587 ms、draft dispatch20,463→19,956。R9700は **447.963519646／24.374921556 tok/s**、TTFT18,323.255／E2E23,533.723 ms、draft dispatch20,361→19,818。target dispatchは各77,406／73,335で不変、p/q53／50block、採用74/106・78/100も不変。HIP-only／fallbackなし／cleanup0。R service復元は同一hash／UUID・healthz/readyz200・接続0。`r40-ordinary-comparison.json`。Vの単回値は改善したが、R500/25は未達で、正式medianやPhase完了に代えない。

### R43 複数行BF16 concatの投入回数削減

- MTP fusion入力は各行のembedding norm／target hidden normをCopy op2個で結合し、8192prefixでは16,384 copy kernelを投入していた。既存Copy ABIはcontiguous同形状のみで、strided／2D copyはない。R40のKV後の不要計算削除でも、このKV前のcopy列は残る。`r43-mtp-fusion-row-copy.md`。
- 一般のBF16 row-major行列2個を列方向へ結合するprivate GPU操作を実装する。行数・左右列数を引数とし、QwenやH5120をnative kernelへ埋め込まず、uint16 load/storeでsigned zero／NaN payloadを含む全bitを維持する。新しい公開C ABIやCopy既存契約の変更は行わない。2D launchでrowとcolumnを割り当て、各要素の64bit除算を避ける。
- 初期のQwen MTP接続は実行M>1だけとし、両RMSNorm producerを既存segmentで完了させてから1回のconcatを呼び、fusion matmulより前に完了させる。M1とcapability非対応は従来copy列を維持する。state投入後の失敗をfallback再実行で隠さない。runtime Mとgraph capacityを混同せず、inactive suffixは書かない。
- native wrapper／kernel／bitwise GPU testとcore／HIP接続を実装し、sourceを凍結した。連続stride、範囲、overflow、BF16整列、outputとinputの重なりを検査し、読み取り専用input同士のaliasは許す。focused host test／core・HIP check／fmtはPASS。モデル実行で実際の採用数を確認できるよう、成功時だけ増える`draft_bf16_row_concat_count`をbenchmarkへ追加した。
- normal HIP／G1／host CMake、Cargo依存、runtime include／unavailable stub、H3 source110件・kernel174件を登録し、manifest hashを同期した。両targetのkernel・runtime compile-onlyとtest strict syntaxはPASS。H3初回はkernel名のソート漏れ1件を検出して修正し、再検査31 tests＋490 subtests PASS。両targetのfresh release buildを開始した。GPU上のbitwise一致、通常モデル経路と速度は未確認であり、新concatによる速度改善はまだ主張しない。

- R43両target release buildはexit0、source before/after一致。V620 fresh archiveのbitwise GPU検査はPASS。通常8192/128もexit0、128token hash `316dc877624f4f03edafe20ff3002ef53dd4f4d90742441bcf23dd61e9ba27ff`・p/q53block・draft74/106採用はR40と一致。既存semantic draft dispatch19,956→3,574に加え、private concat8回を明示記録した（private操作は既存dispatch数に含まれない）。単回prefill221.694／decode24.812 tok/s、TTFT36,990.405／E2E42,109.037 ms、HIP-only／fallbackなし／cleanup0。R40単回219.257／25.143との比較であり、正式中央値やdecode改善は主張しない。V APIとR通常実行は進行中。`r43-v620-output-comparison.json`。

- R9700通常8192/128もexit0、128token hash `7d939adfc7c9b60bb30677bd1c56d462cc6a3b1b86c7512c29913f5d0bb99155`・p/q50block・draft78/100採用はR40と一致。既存semantic draft dispatch19,818→3,436＋private concat8回、target73,335は不変。単回prefill464.906／decode24.900 tok/s、TTFT17,656.051／E2E22,756.567 ms、HIP-only／fallbackなし／cleanup0。R40単回447.964／24.375からの変化は記録するが、500/25目標は未達で正式反復へは進まない。`r43-r9700-output-comparison.json`。

- V620 R43 APIはtext／SSE／MTP進行後cancel／recovery／8192/128の5case HTTP200 PASS。cancel auditはdraft提案2／採用2後のcancelled、続くrequestはcompleted。長入力text SHA `85fef6694c7837ecf682f907e80686ce218c48bf3cc4da94f2a899b4965aa1ea` はbenchmark・R39 APIと一致。終了時current／retryable／quarantine0、VRAM peak32,135,819,264 B、遅延確認でVRAM17,215,488 B／GTT15,020,032 B／busy0へ復帰した。`.local-artifacts/phase83/p835-v620a-r43-concat-r1/`。

### R42 GDNのQ/K共有候補（不採用）

- R19 target prefillのGDN recurrent columnは192 dispatch／集計865.625 ms。trace timestamp異常があるためkernel内訳の順位として使い、walltime上限とは扱わない。llama.cppの同種recurrent処理を参照し、1waveに2列×同じQ/Kを持つ3 value headを割り当ててQ/K loadを共有するscratch候補を作成した。演算式・各列のwave reduction順を維持する意図で、production sourceは変更していない。
- gfx1201 strict compile／linkはPASS。VGPRはcontrol31→candidate59、LDS／spill0。実機ではmixed M1/2/3/63/64/65/2048とzero M3の全output・state bitwise比較、独立long-double oracleはPASS。一方nonfinite M3はoutput515／state15,900 bit不一致（非有限値数は同数）、cleanup検査も155,189,248 B残留でFAILした。終了コード1を全体PASSへ読み替えない。
- M2048 AB/BA各20sampleのkernel中央値は4.303949→5.287839 ms、22.86%遅かったため不採用。性能でも不採用なので数値／cleanup失敗を修正して同じ候補を再計測しない。証拠は`r42-r9700-gdn-multihead-col2-r1/probe-run.log`、scratchは`r42-gdn-multihead-col2/`。元serviceの復元結果は別途lease記録で確認する。

- R43 R9700 concat probeもexact gfx1201でbitwise／境界PASS。`r43-r9700-integration-r1/final.json`はprobe・model exit0、元serviceのhash不変／UUID一致／healthz・readyz200／client0／restore0を記録する。R42 leaseも`r42-r9700-gdn-multihead-col2-r1/final.json`で同条件の復元を確認済み。R42のprobe exit1は保持する。

### R44–R46 次の改善箇所の絞り込み

- R43後もR9700は464.906／24.900 tok/sで500/25未達。prefill walltimeを17.621→16.384秒以下へ縮める必要がある。R43のCPU user+systemはprefill261／decode271 USER_HZ ticks（R40は311／279）であり、GPU計算以外の投入処理も検討対象とする。CPU時間とwalltimeを同一視しない。
- R44は既存の失敗候補とllama.cpp参照を踏まえたNVFP4 ID89／attentionの未試行案1件、R45はcore／frontendの重複処理・同期削減案1件をread-onlyで調べる。R46はsourceを変更せずR43の通常8192/128を1回profileし、R19のtimestamp異常と古いdispatch構成を現在の証拠へ更新する。正式速度の再計測や合格狙いの反復ではなく、timestamp検査と処理内訳の確認に用途を限定する。

- R45調査後、pre-transition長検査の省略案は実装せず、前後両方の検査を維持したままHIP metadata取得を軽くする方針へ絞った。`KvStateResource::snapshot`はnative viewの作成／query／releaseを行うが、既存`sllm_kv_state_query`で同じpublished metadataをregistry／accounting lock下に取得できる。直接queryへ置き換え、identity・generation単調性・physical metadataの検証を維持する。readback用のnative viewは保持する。linear-attention側は既に直接queryであり、初期調査の「64state全てでview allocation」という見積もりは誤りだったため、対象を16 KV layerへ訂正する。新しいnative ABIや数値変更は不要。

- R44 rsqrt hoistはgfx1201 control／candidateのISAで比較し、controlが既にkey loop外へ1個の`v_s_rsq_f32`をhoist済みと判明。命令削減なし、WaveLocal側VGPR95→101・occupancy16→12へ悪化するため、GPU測定せず不採用。`r44-target-next-candidate.md`。

### R47 R36 attentionの通常経路への接続修正

- R46 fresh R43 profileはbenchmark exit0、出力・MTP採否一致、80,151 dispatchを記録した。既存analyzerはR32後の`final_support_v1`を認識せず失敗したため、その失敗とraw traceを保持し、kernel名別の集計を別途保存した。元serviceのhash／UUID／health／client確認後の復元はPASS。
- rootが通常traceを確認すると、long-prefix Q8/W16は48回すべて`<6u,6u,false>`で、R36のWaveLocalKv=trueは0回だった。host launcher内の`#if defined(__gfx1201__)`が原因で、host-only macro dumpでも当該macroが未定義と確認した。R36 scratchの速度改善はそのscratchの証拠として保持するが、従来の統合数値PASSとR43までの通常生成を「最適化variantが実際に選択された証拠」とは扱わない。
- runtimeのexact GPU targetとM2048条件をprivate launcherへ明示的に渡す修正を行う。公開ABIや数値計算は変更しない。fresh binaryの数値検査に加え、実traceでWaveLocalKv=trueが選ばれたことを確認する。他のhost launcherにも同じdevice macro誤用がないか、対象を絞って確認する。

- R45 metadata直接queryとR47明示runtime boolを合わせたfresh release buildは両target exit0、source before/after一致。H3は31 tests＋490 subtests PASS、fmt／Markdown link／diffもPASS。R47 sourceは`arch_name == gfx1201 && query_count == 2048`でWaveLocalKvを選び、device専用macroをhost launch判断に使わない。他のhost routingの限定auditでは同種の誤用は追加で見つからなかった。通常モデルとR47 variantの実機確認を開始した。

- R47 fresh production archiveに対するfalse／true比較は、M2048・prefix2048/4096/6144で全出力bit一致とsampled independent oracle、M129境界でfull oracleをPASSした。通常V620は220.804／25.237 tok/s、R9700は469.952／24.764 tok/sの単回値で、両方R43と128token・MTP採否・dispatch数が一致、HIP-only／fallbackなし／cleanup0。R45単体の速度改善はこの複合候補から分離して主張しない。`r47-v620-output-comparison.json`／`r47-r9700-output-comparison.json`。
- R9700の通常経路をprofileし、Q8/W16の`WaveLocalKv=true`48回・false0回を実kernel名で確認した。R43のfalse48回・true0回から選択が変わり、同じ128token hashを維持した。`r47-r9700-dispatch-profile-r1/wave-local-dispatch-counts.json`。profileの時間は正式速度へ使わない。500/25未達のため最終反復・公開は残る。

- R47 leaseはprobe／ordinary／profile／restore全てexit0、元serviceのfrozen input hash不変・exact UUID一致・healthz／readyz200・client0を確認した。`r47-r9700-integration-r1/final.json`。次のR48はID89のFP8 operand LDS row strideとbank競合の関係をscratchで調べる。64→68 byteのpaddingはload alignmentの適合を確認してから判断し、alignmentを満たさない配置を速度目的で使わない。現時点ではproduction変更／GPU実行なし。

### R48／R49 LDS配置の独立候補

- R48はFP8 operandのLDS row padding、R49はFP32 scale tableの物理stride4→5を別々のscratch候補として比較する。R49はlogical scale_blocks_per_stage=4と全FP32値・K16 WMMA・Kahan順序を維持し、配列の物理間隔だけを変更する案。想定LDS増分は各1,536 Bで、30720→32256 Bとなるが、実際のcodegen／alignment／occupancyとGPU計測で採否を決める。
- llama.cpp pinned `ggml/src/ggml-cuda/mmq.cuh`のshared-memory padding方針（32–33、114、NVFP4 stride assert159行）を構造上の参照とする。CUDA/MMQ側のalignmentをrocWMMAへ無条件に転用しない。source copyや数値型変更は行わず、productionはR47のまま保持する。
- 同時にR47 R9700の通常APIを再検証する。直近R19以降のp/q・metadata query・attention routingを含む5caseのHTTP／SSE／MTP進行後cancel／recovery／8192/128、cleanupと物理memoryの範囲を確認する。正式速度の反復とは分ける。

- R49の初期bank計算は32-bankを仮定していたが、[AMD HIP公式のbank conflict説明](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.3/understand/performance_optimization.html#bank-conflict-theory)はRDNA2以降を64 bank／4 Bとしている。したがってscale stride4で競合があるとの初期見積もりを撤回し、64-bankのlane／instruction mappingで再確認する。既存配置で競合がなければR49はGPU実行前に止める。R48も同じhardware前提で見直す。

- R47 R9700 APIの5caseは全てHTTP200 PASS。cancelはdraft提案2／採用2後にcancelled、次requestでcompletedへ復帰。8192/128はdraft78/100採用、text SHA `376acf36219f44832c8a72e4fdb90cabd41f6db1f0facb015a2adfc72c8adfb5` が通常benchmarkと一致。終了時current／retryable／quarantine0、VRAM peak32,253,505,536／total34,208,743,424 B、GTT baseline266,579,968→peak266,657,792 B。`.local-artifacts/phase83/p835-r9700-r47-dispatch-r1/execution.json`。元service復元の最終状態はlease記録で確認する。

- R47 R9700 API leaseの`final.json`を確認し、API／runner／restore exit0、元serviceのfrozen input hash不変、exact GPU UUID、healthz／readyz200、接続0への復元を確認した。agent報告文中のserver hashには転記誤りがあったため、identityにある`7ca68ed192f18e843945650c2222e6c647d47fe5453a99da30cd7faddb9a3e8a`を正本とする。
- R49 scale stride5は実装前に不採用。64-bankではweightの16列は各subreadで異なるbank、activationの2種類のrowも32bank離れ、想定した競合がない。既存ISAはscale readを既にまとめており、stride変更の追加LDS1,536 Bに見合う利点を特定できなかった。`r49-scale-stride5/README.md`。production変更・compile・GPU実行なし。
- R48 stride68のgeneric wrapperはstrict gfx1201 compile PASS、VGPR228→242／LDS30,720→32,256 B／報告occupancy6→5。ただしroot reviewで、これは実M2048のaligned StaticK/StaticN wrapperとの比較ではないと確認した。genericの悪化だけで通常ID89の候補を棄却せず、実際のK5120/N17408と逆tupleのaligned control／candidateで追加compileを行う。数値型・演算順を変えないscratch調査で、まだ数値PASSや速度改善の証拠はない。
- R50はhost／runtimeの重複同期・allocationをread-onlyで調べる。R39・R40・R43・R45の既存削減やstate長検査省略を繰り返さず、残る具体的な呼出しとlifecycleを確認してから候補を決める。

### R51／R52 演算順を保つ次候補の調査

- R51は実際に選択されるgfx1201 Q8/W16 WaveLocalKv版のload／index／schedulingをread-onlyで調べる。既に不採用のsoftmax block化・FMA・rsqrt hoist・W24を再試行せず、現行ISAで未削減の処理がある場合だけ候補化する。
- R52は`e2m1x4_to_e4m3fn_exact_bits`の4bit→byte lane展開を、4項のmasked shiftから2段のbit spreadへ置き換えるscratch案。positive lookup、sign bit、FP8値、WMMA／Kahan順は維持する。llama.cppのpacked integer lookupを構造上の参考とし、新しいdtypeや丸めを導入しない。rootの独立host検査では全65,536 uint16入力で従来展開・nibble別oracle・E4符号付きbyte列が一致した（`r52-root-expansion-check.json`）。これはGPU builtinの正しさや速度の証拠ではなく、tiny kernelの命令数と実aligned ID89のresource比較へ進む根拠とする。

- R48 aligned両tupleもstrict compile PASS。control46 SGPR／212 VGPR／30,720 B LDS／報告7 wavesに対し、pad68は40／224／32,256 B／6 waves、spill0だった。rootは、このresource差だけでは8-wave workgroupとLDS制限を含めた実際の同時実行数・速度悪化を断定できないと判断した。従ってcompile時点での不採用にはせず、exact aligned controlとの数値・AB/BA速度比較を行えるscratch harnessを準備する。bank競合の改善も未証明であり、採用は実測後に決める。
- R52はroot側で実device codecの全入力・65535/65536/65537の境界・guard・repeatを検査するstandalone harnessを作成し、strict gfx1201 compile/link PASS。最初のbuildは`-x hip`不足によるunused GPU optionで失敗し、source言語を明示して修正した。まだGPU未実行であり、host exhaustive PASSと区別する。`r52-root-gpu-probe/identity.json`。候補のcodegen改善が確認できれば既存のR9700 lease手順で検証する。

- R52のtiny helperは0x8c→0x6c byte、full TUの実aligned ID89両tupleは0x3094→0x2ebc byte（472 B減）。SGPR46／VGPR212／LDS30,720 B／spill0は不変。命令列が短くなった証拠として採用し、正確な速度改善率とは扱わない。旧版と候補を同じ実行内で測るaligned matmul harnessを準備する。productionのhelperはまだ変更していない。`r52-e2m1-swar/r52-summary.md`。
- R50は既存R8で低優先度にしたfull-accept時の`last_output.clone()`省略を再確認した。現行device経路のhidden幅は5120、3行BF16で30,720 B。50 blockすべてfullと仮定しても約1.5 MiBのhost copyで、R9700 decodeで残る約48 msを解消する根拠は弱い。state管理APIを増やす変更は今は実装せず、候補として保留する。agentの初期幅2560／15,360 B表記を訂正する。918 physical fenceの一括削減も、owner／state publicationとの依存を外す根拠がなく実装しない。`r50-mtp-full-commit-output-owner.md`。

### R48／R52 実機検証とR48本体への接続

- root reviewでR48 harnessのcontrol分岐が再びgeneric lookaheadを呼ぶことを検出し、GPU実行前に修正した。aligned M2048は実productionのaligned control2種、その他は既存genericを呼び分ける。M2048のNaN fixture2件も追加し、candidate自身のnonfinite経路を検査した。比較対象の訂正前にはGPUを実行していない。
- `r48-r52-r9700-scratch-r1`はexact gfx1201／UUID固定で両probe exit0。R52 codecは全65,536入力と65535/65536/65537境界、guard、両variant反復をPASSした。これはcodecのGPU証拠であり、R52 matmul全体の速度検証は別に進める。
- R48のM2048両tupleは全finite出力pair差0、repeat PASS、sampled long-double oracle最大0 ULP。NaN fixtureもcandidate自身の分類／repeat／oracleをPASS。M128/127/129の両tupleはcontrolのみの境界検査で、paddingの数値証拠と混ぜない。全10case PASS。
- R48の20 AB＋20 BA sampleをまとめた中央値は、K5120/N17408が13.900493→11.018881 ms（約20.73%短縮）、逆tupleが13.918014→10.988722 ms（約21.05%短縮）。AB／BA個別でも改善した。NaN fixtureでも改善したが通常速度評価はfinite値を使う。resource増加だけで候補を棄却せず実測した結果であり、全モデルの改善率やbank競合の直接測定とは扱わない。`root-audit.json`にraw logとの対応を保存した。
- 同leaseはrunner／restore exit0、元service hashとunit不変、exact UUID、healthz／readyz200、client0を確認した。root session20905はterminal0。R52 full matmulの凍結済み比較は別の単独leaseへ引き継ぐ。
- R48本体実装はgfx1201・M2048・既存2 tupleだけに限定する。共通lookahead bodyにpadding templateを加え、既存M／fallbackはunpaddedを維持する。device symbolと実host dispatchを両方更新し、公開GPU検査に2047/2048/2049、host selectorに隣接値とその他aligned Mを追加する。H3 inventoryへ追加するkernelは2個。fresh build／公開GPU／通常モデルの計測はまだ未完了。

- R52 full matmulも`r52-r9700-matched-r1`でexact gfx1201 PASS。M256/1024/2048・2 tupleのcandidate8caseはfinite全出力差0、非finite分類／repeat一致、sampled oracle最大0 ULP。M257の2caseはcontrol-only。pooled40のcontrol/candidate比は1.0130〜1.0160、M2048 wide13.510597→13.316457 ms、down13.533581→13.323919 ms。probe／runner／restore exit0、frozen hash不変／UUID／healthz・readyz200／client0を確認した。
- rootはR48とR52を組み合わせた本体で次の通常計測へ進む。R52は共通のgfx1201 exact ingress helperだけの整数展開変更で、GPU全uint16検査によりFP8 byte列の同一性を確認済み。個別scratchの改善率を足してfull-model速度を予測したり、未測定の全shapeの高速化を主張したりしない。
- R51はbyte-safe scale bulk loadのscratch compileで、16 scalar byte loadを2 global64 loadへ減らした。一方VGPR95→100、load wait数31→44、コードサイズ14,668→15,044 Bへ増えた。これは資源上のリスクであり実際の速度悪化の証拠ではない。実測改善済みのR48/R52を優先し、R51は本体へ採用せず保留する。ASan/UBSan4,544境界検査はhostのbyte範囲の証拠で、GPU数値PASSとはしない。`r51-mxfp8-scale-row-pack.md`。

- R48+R52のsourceを凍結し、両target fresh release buildはexit0／source before-after一致。V benchmark SHA `181976ca63c25a7654af117e268ff399eab9b86911d8eb970c224d44b734c325`、R `fa80b1e11f3451360ec56f2a8bf1e8070ab407614824256b2c3a6578be97b545`。rootは共通bodyの全buffer store／leading dimension、exact M2048のruntime dispatchとmetadataの対応を確認した。
- host selectorの境界検査はPASS。初回GCC strict compileのrange-loop-copy警告はconst参照へ修正した。公開GPU testは両target設定のstrict syntax PASS、Rのfresh archiveにリンクしたprobeもbuild exit0。
- H3は初回kernel総数174→176、次回追加集合71→73の古い期待値を検出した。2回目は30 tests＋490 subtests PASS、修正後は失敗した1testだけに絞りPASSを確認した。全体を繰り返す検査は止め、同じsource範囲の累積結果として扱う。これらCI testの期待値修正はbuild source identityの対象外で、native／Rust sourceはビルド中に変更していない。
- fresh V620通常生成を`r48-r52-v620a-ordinary-r1`で開始した。R9700は`r48-r52-prefill-public-gfx1201-r1`の公開検査後、`r48-r52-r9700-ordinary-r1`で通常8192/128を測定する。現時点では新本体の速度目標達成を認定していない。

- R48+R52 fresh V620通常はexit0、prefill219.127331133／decode25.316564164 tok/s、TTFT37,422.816／E2E42,439.375 ms。128token hash・MTP採否とdispatch数はR47と一致し、HIP-only／fallbackなし／cleanup0。`r48-r52-v620-output-comparison.json`。このidentityで1 warmup＋3 measuredの正式測定を開始した。
- R9700 integration r1はpublic wrapperがbuild済みprobeと異なるlabelを渡し、`b.is_file()` assertionでGPU投入前に終了した。モデル測定は未開始（exit125）。kernel不具合や数値FAILとは扱わず、この起動失敗を保持する。元serviceはhash不変／exact UUID／healthz・readyz200／client0／restore0へ復元した。build directoryと同じpublic labelへ修正し、cwdも明示してr2へ進む。source変更・再buildは不要。

- 修正r2のR9700公開plan/executeは、2 tuple×M255/256/257/2047/2048/2049の12caseをPASS。M2048のdevice metadataは新pad68 symbol、隣接Mはgenericを実際の公開実行結果で確認した。全finite／repeatとsampled boundary oracle最大0 ULP、resources_released=1。`r48-r52-prefill-public-gfx1201-r1/gpu.log`。
- R9700通常測定の実labelは`r48-r52-r9700-ordinary-r2`（r1では未実行）で、PID2294203が実行中。V620正式1+3は`r48-r52-v620-formal-r1`、PID2288671／root session97088が実行中。途中状態を速度PASSや解放完了へ読み替えない。

- R48+R52 V620正式1 warmup＋3 measuredはexit0。中央値prefill215.468671313／decode24.808386174 tok/s、MAD0.177741597／0.026056350。TTFT38,050.128／E2E43,206.282 ms。rootがmeasured3回から中央値／MADを再計算し一致を確認、warmupを含む全4回128token hash一致、HIP-only／fallbackなし／cleanup0を確認した。現identityで200/20を満たす。`r48-r52-v620-formal-summary.json`。
- R9700通常r2もexit0、prefill517.648134665／decode24.719682580 tok/s、TTFT15,861.534／E2E20,999.205 ms。R47から128token hash／MTP採否／dispatch数は維持し、HIP-only／cleanup0。prefillは単回で500を超えたが、decode5137.606 msは25 tok/sに必要な5080 msを約57.606 ms超えるため、合格狙いの正式反復をせず追加改善を続ける。`r48-r52-r9700-output-comparison.json`。元serviceはhash／UUID／healthz・readyz200／client0／restore0へ復元済み。
- 次はdecodeに限定して調べる。R53は実M2/3でのNVFP4 gate/up入力量子化共有、R54は未削減のhost操作／小コピー、R55は既存ResidualRmsNorm融合の適用条件を対象にする。rootは現行guardがQwen3.5-4B fingerprint／layer数へ限定されることを確認したが、Qwen3.8へ広げてよいとの判定はまだしていない。構造上の条件・BF16中間丸め・既存の不採用理由を調べてから実装を決める。

### R53 MTP verify時のNVFP4 gate/up量子化共有

- 調査で、coreの共有predicateが実token_count<64を除外し、MTP width2のM3 target verifyでNVFP4 gate/upが分解経路へ戻っていると確認した。既存graphは56個のNVFP4 gate/up packについて入力とinput-global scaleの一致を保証する。native共有runtimeには「量子化1回→gate→up」の処理が既にあり、default ID94のM2〜4 memberを受け入れるvalidation／workspace／grid対応が不足している。`r53-nvfp4-mtp-gateup-activation-sharing.md`。
- 実装範囲を実M2/3/4の共有への拡張に固定する。既存ID94を使い、量子化byte／scale／member計算順を維持する。M1とその他Mは既存経路を保ち、graph容量で実行Mを判定しない。Unsupported時のprepare前fallbackだけを許し、投入後の失敗を再実行で隠さない。hostと公開GPU検査でM1/2/3/4/5の境界、dispatch metadata、数値・解放を確認してから通常生成を測定する。
- topology上はR9700の50 verify block×56 pairで2,800 quantize launch削減が見込まれるが、通常auditで確認するまでは実測削減数としない。prefill・sampling・companionのBF16経路や新しい公開ABIは変更対象にしない。
- core／HIP Rust wrapper／native runtimeへ実装し、CPU-onlyの公開runtime host testをPASSした。両target設定でM2/3/4・M64以上のprepare／execute、M5/63のUnsupported、workspace=M×2880 B、dispatch_count=3、queue scratchの拡張と再利用・解放を確認した。coreのprojection-pack関連3testとshared activation関連5test、HIP wrapper境界1testもPASS。rootは実token_countによるviewとprepare時のみのfallback、既存quantize→gate→up順の維持を確認した。これはGPU数値・通常生成・速度の証拠ではなく、公開GPUテストの境界追加とfresh buildへ進む段階である。
- 続いて両targetのrelease buildをPASS。build中の差分はstandalone公開GPU testだけで、本体・archiveのbuild入力は不変と確認し、別コンパイルした最終test sourceとの対応を`r53-build-source-mapping.json`へ記録した。関連H3検査は4test＋335subtestをPASS。
- fresh archiveからの公開GPU検査も両targetでPASS。M2/3/4の行別入力6/3/1.5/0、member ID94・grid544・workgroup256、共有quantize grid=M×40／workspace=M×2880／dispatch3、全出力analytic oracle／directとのbitwise一致／repeat／解放を確認し、M5のprepare Unsupportedと既存M1／M64以上を維持した。`r53-pack-public-gfx1030-r1`／`r53-pack-public-gfx1201-r1`。両GPUの通常8192/128測定を開始し、速度・実dispatch削減は結果待ち。
- 通常生成も両GPUで完了した。V620は218.997822726／25.116535896 tok/s、128token／MTP採否一致、target77406→74438（2,968減）、draft3574維持、HIP-only／cleanup0。R9700の結果は次のR55項へ記録する。`r53-gfx1030-output-comparison.json`／`r53-gfx1201-output-comparison.json`。R9700 leaseはpublic／model／runner／restoreすべてexit0、元service／binary hash一致、UUID一致、healthz・readyz200／client0へ復元済み。R53での正式反復は行わず、R55後の候補へまとめる。

### R54／R55 後続候補の調査

- R54はcompanionの中間hidden rowをGPU上に保持する案。現行width2・50 blockでは各10,240 BのH2DとD2Hが100回ずつあり、初回H2D50回と同一queueのD2D50回へ置き換えるとhost転送150回・1,536,000 Bを削減できる計算になる。実装・実測は未実施。別buffer範囲、queue順序、copy寿命、失敗時のstate処理が必要なため、まず既存共有経路だけで済むR53を測る。`r54-mtp-hidden-handoff.md`。
- R55は既存ResidualRmsNorm融合の適用拡大案。現在はQwen3.5-4Bの32 layerに限定されるが、Qwen3.8の64 layerにも隣接したattention residual addとpost-attention RMSNormが各1組ある。既存融合はadd後のBF16丸めを維持し、別演算と同じRMS reduction順を使う。過去の不採用理由は見つからず、従来の検証範囲による限定と判明した。ただしH5120での実機bitwise比較、artifact／sidecarの一致、projection packとの併用確認は未実施であり、本体への適用はR53測定後に判断する。調査からの速度推定を実測として扱わない。 `r55-qwen38-residual-rmsnorm-expansion.md`（SHA-256 `2bbcec60bc707e538639e7a50a9b7ae8f9e75d3b400f6f6d94cf0a52953b3c61`）。

### R55 既存ResidualRmsNorm融合のQwen3.8適用

- R53 R9700通常結果はprefill515.376911708／decode24.923309823 tok/s、decode5095.631395 ms。出力128tokenとMTP採否はR48+R52と一致し、target dispatch73335→70535の2,800減を確認した。HIP-only／cleanup0。まだ25 tok/sに15.631395 ms不足するので正式反復へは進まず、R55を実装する。R54は保留する。`r53-gfx1201-output-comparison.json`。
- 実装範囲は既存のResidualRmsNorm演算・graph rewriteの再利用とし、新kernel／公開ABIは追加しない。既存4B条件を維持したうえで、verified Qwen3.8 artifact／recipe digest一致、27B fingerprint、64 layerの3-linear/1-full schedule、text target／adapterなしの既存検証範囲へ広げる。MTP companionは変更しない。
- 受入条件は、64 pairの融合と既存projection packの共存、残差中間BF16値とnorm出力の保持、同じFP32 add・BF16丸め・RMS reduction順、既存rollback設定の維持。H5120のM1/2/3で両GPUの既存numerical oracleを拡張する。Qwen3.8のOffsetOne（1＋重み）設定と既存Direct設定を検査し、残差中間はCPUの明示BF16丸め、norm出力は通常GPU RMSNormとのbitwise比較でも確認する。既存CPU oracleの逐次FP32和とGPUのwave reductionは別の丸め順なので、CPU比較だけをN0の証明にしない。通常8192/128の出力・採否・dispatch・cleanupと速度も確認する。非有限入力の未測定bit一致やBF16 full-model品質同等性は追加認定しない。
- 初回V620公開test r1はH5120・M1・OffsetOneのCPU逐次FP32 oracleに対し1 BF16 ULP差でFAILした。中間BF16は一致していた。rootの独立再現ではrow0の逐次和12009.392578125がexact long-double和12009.470058441162からずれ、col589のBF16中点を跨いでbfa7となった。一方wave和12009.4697265625はbfa6となり、実GPUと高精度参照側に一致した。全5120列×3row×両scale modeもwaveと高精度参照のBF16差0、逐次和ではOffsetOneのrow0に19列／row1に12列の差があった。`r55-root-oracle-diagnosis/`。
- r1失敗を保持し、scratch診断ではCPU逐次差を記録したうえで通常GPU RMSNormまで実行した。V620のH5120・M1/2/3・Direct／OffsetOneすべてで、融合中間はCPU BF16 Add、融合norm出力は通常GPU RMSNormとbitwise一致し、解放後のVRAM／GTTも元の17,215,488／15,020,032 Bへ戻った。`r55-rmsnorm-diagnostic-gfx1030-r1`。正規testのH5120数学oracleを高精度積和へ修正し、厳密BF16比較とGPU pair一致を維持して再確認する。native数値kernelと許容差は変更しない。
- 修正した正規test r2はV620でPASS。baseline11case／residual10caseを実行し、H5120の6caseは高精度数学oracleと通常GPU RMSNormの両方に対するbitwise一致を確認した。`r55-rmsnorm-public-gfx1030-r2`。R55ではnative数値sourceを変更していないため、R53 archiveを再利用し、native／include等129ファイルが不変である対応を`r55-native-evidence-mapping.json`へ記録した。core統合後buildとの最終archive対応とR9700実機検査は残る。
- core実装とfocused host5testをPASSした。既存4B public rewriteを維持し、共通rewriteへQwen3.8のartifact／digest／64-layer schedule／851 weight binding条件を追加するcrate-private経路を接続した。64 fused pair、104 NVFP4／FP8 GDN projection packとの共存、異なるartifact・schedule・MTP／multimodal・不足weight・rollbackの検査を含む。初回の未使用import警告はtest側への移動で解消した。`r55-implementation/`。rootは元snapshotとの差分を確認し、既存R53テストの整形だけを戻した最終sourceから両targetのrelease buildを開始した。
- 両target buildをPASSし、source before／after／current一致とnative archive不変を確認した。R9700公開r2もbaseline11／residual10caseをPASS。通常R9700はprefill518.062234330／decode24.790391922 tok/s、decode5122.952489 ms。R53と128token・採否が一致し、target70535→67079（3,456減）、draft3436維持、HIP-only／cleanup0。ただし単回decodeはR53より遅く、速度改善は未確認。正式反復には進まず、次のR54転送削減へ進む。`r55-gfx1201-output-comparison.json`。
- V620通常も220.878246744／25.073080132 tok/s、128token／採否一致、target74438→70790（3,648減）、draft3574維持、HIP-only／cleanup0。`r55-gfx1030-output-comparison.json`。R9700 leaseも全exit0、元service／hash／UUID一致、healthz・readyz200／client0へ復元した。API準備は既存r39 runnerで保存し、R55の正式反復／API実行は追加せず次候補へまとめる。

### R54 MTP companion hidden行のGPU内引渡し

- R55まででdecode25 tok/sは未達のため、調査済みのhidden転送削減を次の実装対象にする。通常width2で、各blockの最初のtarget hidden uploadは保持し、companionの中間hiddenを同一session／queue上に保持して次のproposalへD2Dで渡す。最終proposal後の未消費hiddenはCPUへ戻さない。targetのhidden管理・token selector readback・p/q／RNG・state commit／rollbackは維持する。
- 既存range-checked DeviceCopyとその寿命管理を再利用し、新native kernel／C ABIは追加しない。source／destinationは別の非重複範囲とし、BF16 shape／stride／session identityを確認する。非対応の場合は要求を変更する前だけ従来host経路へ戻し、copy／graph失敗後は既存のpoison／cleanupへ従う。通常decodeへ戻るときやabort／reuse時に保存したdevice hidden状態を持ち越さない。
- 受入条件は、width1/2/3境界、full／partial accept、次proposal入力・中間hiddenの意味、copy失敗時の寿命／解放、通常経路の保持。両GPUの通常8192/128で出力・採否・HIP-only・cleanupと実転送削減／速度を確認する。width2・50blockではhost転送150回と1,536,000 B削減が見込まれるが、実測前に速度改善を認定しない。R55の融合は数値・dispatch検証済みの候補として維持し、その単回結果を高速化の証拠へ置き換えない。

- R54の途中実装をrootで確認した。既存DeviceCopyはqueue／source／destinationのsessionとrange、非重複を検証し、HIP transfer側も両bufferを保持する。新しいchain経路の通常生成復帰時のcache破棄と、未対応backendに対する要求変更前の従来経路選択は実装担当へ確認中。copy後のgraph失敗とcopy自体の失敗を試験結果で区別する。まだsource凍結／host試験完了／実機PASSとは扱わない。
- R54実機比較scriptを準備した。R55との128token／text hash、MTP採否、target／draft dispatch、target kernel選択、HIP-only／cleanupを照合する。host転送削減量はblock数からの予測と実観測を分ける。R9700 wrapperは元service復元を維持し、通常結果の比較と速度閾値を満たす場合だけ正式反復へ進む。準備時に再混入したHOME上書きを検出して削除し、通常結果の比較失敗でも元serviceを復元してnonzero終了することを確認した。wrapperのbash syntaxと比較scriptのPython syntaxはPASS、GPU実行は未開始。`.local-artifacts/phase83-5/r54-compare.py`、`r54-run-preparation/`。

- R54をsource freezeした。core3test（width1/2/3の転送契約、chain開始前拒否、copy失敗、copy後graph失敗）とfrontend Qwen MTP6test、既存fixed-K20 all-accept／partial-rejection parser検査、cargo checkをPASS。実際の数値演算を行わないrecorderの結果はhost契約に限定する。frontendはfixed-K20 p/qの独立RNG経路だけでchainを使用し、その他selector経路はhost hiddenを維持する。rootは最終diffとsource hashを照合した。core SHA `962243bb6c63097f5b541080469bc81f886d22d69db1c7e328bc339cb13c47eb`、frontend SHA `6ff15d8a4084b8680dea694e12203652f0cd58393f4a45356eff69530162010c`。
- 両targetのfresh release buildを `candidate-gfx1030-r54-device-hidden-r1`／`candidate-gfx1201-r54-device-hidden-r1` で開始した。数値kernelのsourceはR55から変更していない。native証拠はsource／archive対応を確認して再利用し、今回変えたruntime経路の実GPU出力・採否・cleanup・転送／速度は別に測る。

- R54両target buildはexit0、source before／after／current一致。native archiveもR55と同じで、対応を `r54-native-evidence-mapping.json` に保存した。通常8192/128はV620 220.087363696／22.076790375 tok/s（decode5752.647819 ms）、R9700 511.469502362／24.850342457 tok/s（5110.593555 ms）。両GPUのtoken／text hash、MTP採否・dispatch・target kernel選択はR55と一致し、HIP-only／cleanup0。R9700 leaseは通常／比較／runner／restoreすべてexit0、元service hash／unit／UUID不変、healthz・readyz200／client0へ復元した。
- V620はR55単回5065.193400 msから約687 ms遅く、R9700も25 tok/sに30.593555 ms不足した。R54を速度改善済みとして採用せず、正式反復は行わない。V620のruntime／memory-copy traceを `r54-v620a-transfer-profile-r1` で開始し、小さいD2D転送と同期の遅延を調べる。profile値を通常速度の代用にはしない。別途R56候補としてMLP residual add→次layer input RMSNorm／最終RMSNormへの既存融合適用をread-only調査する。

- V620転送profileはexit0、通常R54とtoken／text hash・MTP採否一致、HIP-only／cleanup0を確認した。解放後VRAM17,215,488／GTT15,020,032 Bに復元。profile時decode23.733513792 tok/sは観測負荷が異なるので通常値と比較して改善認定しない。memory-copy CSVにはsize列がなく、JSON／runtime correlationでサイズと方向を追加解析する。`r54-v620a-transfer-profile-r1/output-comparison.json`。

- R54追加確認で、private chain開始後に一般の`decode`系APIへ移った場合、`target_hidden=None`が保存済みsourceを暗黙に消費し得る点を発見した。従来のhidden必須契約を維持するcorrectness修正とhost回帰試験を実装担当へ依頼した。実測済みR54 binaryは変更せず保持する。この修正を未実施のままreleaseしない。

### R56 MLP residualと次段RMSNormの融合候補

- exact Qwen3.8 targetには、layer0〜62のMLP residual add→次layer input RMSNormが63組、最終layerのMLP residual add→final RMSNormが1組ある。既存ResidualRmsNormの2出力（丸め済みBF16 residualとnorm結果）を保持すれば、次layer residualとMTP target hiddenの両消費を維持できる。既存不採用の記録はまだ見つかっておらず、元matcherのattention限定が適用範囲を決めていた。read-only調査中で速度効果は未認定。
- 実装する場合の受入条件は追加64組を対象とし、R55の64組との共存、exact artifact／schedule等の適用条件と既存4B経路の維持、BF16 Add中間・reduction順・scale mode／epsilon・norm結果・依存関係の保持。最終融合は`final_rmsnorm`として識別できるようにし、hidden取得はresidual出力、final norm取得はnorm出力を参照する。中間出力を省略したりMTPへ別の値を渡したりしない。新native kernel／ABIは追加しない。
- 変更範囲に対応するgraph・hidden accessorのhost検査、両GPU通常8192/128の出力・採否・dispatch・cleanup・速度比較を行う。既存kernelの不変な数値証拠はsourceとarchive対応で再利用し、今回のgraph接続とfull-model出力は新candidateで検証する。調査memoを確認してからgraphとcoreの非重複所有で実装へ進む。

- R54のordinary API guard修正はcore focused4test／MTP関連33test（3ignored）をPASSした。ただしR54自体は速度改善未確認・V620単回低下のため本体へ不採用と決定し、guardを含むR54専用経路を削除した。prototype source・diff・測定・修正検証をscratchに保持し、R56 core変更だけをR55 snapshotへ適用した。`r54-removal/removal.json`。R54のD2D自体が原因で遅いと実証した、という結論にはしない。
- traceの汎用集計はCSV／JSONの重複表現と親objectを混ぜるため物理コピー数の証拠として不適切と判断し、時間制限によらず停止した。canonical memory_copy arrayの1,033件はCSVと一致し、明示D2D recordと10,240 B recordはない。copyBuffer kernelは全stream合計3,446件で、stream2のBF16 companion候補958件の実kernel時間は計2.848 ms。API引数がないためhidden行の53 D2Dへ確定帰属せず、R54低下原因は未確定とする。`r54-v620a-transfer-profile-r1/detailed-transfer-analysis.md`。
- R56 read-only memoは追加64組全部の適用条件を確認し、pinned llama.cppのQwen35 graph接続も照合した。新source importなし。graph実装は`qwen_graph.rs`、rootは`qwen_execution.rs`の2つのfinal hidden accessorを担当する。rootはstandalone/fusedから論理的なpre-norm／normalized tensor IDを選ぶ共通helperと、1／3行のtarget hidden・embedding参照を確認するtestを追加した。graph fixtureの追加後に統合host検査を行う。`r56-mlp-residual-rmsnorm.md`（SHA `4cd3988db3b6c97788f12e8faa4d6812e353fc4b757f7115ce3deef9274afd45`）。

- R56 graph／coreをfreezeした。Qwen3.8 graph3testはattention64＋MLP64、2出力・依存remap・final identity・projection104共存・不正pair／scopeをPASS。既存4B graph1testもPASS。coreの新final hidden testは1／3行でpre-normとnormalizedの元tensor ID維持・readbackをPASSし、既存embedding boundary1testとMTP target prefill2testもPASSした。初回core試験はQwen38 synthetic fixtureへ4B用ProvisionSourceを使ったため失敗し、既存ProjectionPackTestProvisionSourceへ訂正して通過した。実kernel数値FAILとは扱わずログを保持する。
- graph SHA `caefa2c70bda09446e6d320b766887157406011660932833739891a7a4f437d6`、core SHA `8a8b5318cc09ca672be8b870e4742c011a5a2bfea47afc7b11c401e1d9498d06`。両targetのfresh buildを `candidate-gfx1030-r56-mlp-residual-rmsnorm-r1`／`candidate-gfx1201-r56-mlp-residual-rmsnorm-r1` で開始した。通常比較はR55を基準に追加64×（prefill4回＋verification block数）のtarget dispatch削減、出力・採否・HIP-only／cleanupを検査する。R56のGPU速度／数値結果はまだない。

- R56両target buildはexit0、source before／after／current一致、native archiveはR55と同じ。通常8192/128はV620 prefill221.066426998／decode24.995771665 tok/s（decode5080.859343 ms）、R9700 520.898265787／24.435369700 tok/s（5197.384020 ms）。R55と128token／text hash、MTP採否が一致し、target dispatchはV70790→67142、R67079→63623で期待した追加64×評価回数の削減を確認した。draft dispatch維持、HIP-only／cleanup0。`r56-gfx1030-output-comparison.json`／`r56-gfx1201-output-comparison.json`。
- R9700 leaseはordinary／compare／runner／restoreすべてexit0、元unit／input hash／UUID不変、healthz・readyz200／client0へ復元。decode目標未達なのでformalはskipした。R56は数値・接続とdispatch削減の検証済みcandidateとして保持するが、速度改善済みの最適化とは認定しない。

### R56後の作業手順の見直し

- R55／R56のdispatch削減が通常decode速度へ結び付いていないため、推測で融合範囲をさらに増やす作業を止め、現candidateのR9700 decodeをprofileする。次の変更はdecode区間のkernel時間、stream間の関係、API時間とkernel間gapから対象を選ぶ。gapだけをCPU待ちやGPU同期の原因へ断定しない。
- ordinary計測を目標達成まで繰り返すことはせず、追加の1回はruntime／kernel／copy traceによる診断に限定する。生成結果・採否・cleanupを同じR56通常結果と比較し、元serviceを復元する。profile throughputを通常性能の代用にはしない。既存wrapperを基に`r56-decode-profile-preparation/`を準備中で、診断はまだ未実行。

### 2026-09-10 ユーザー承認による速度目標の緩和と完了作業

- ユーザーは速度追求を終了し、それ以外の残件を完了するよう指示した。旧200/20・500/25を達成済みとは扱わず、目標変更として保存する。
- 最終candidateはR56。通常単回V221.066426998/24.995771665、R520.898265787/24.435369700 tok/s。R decodeは旧25目標に約2.3%不足する。新しい速度探索は行わない。
- R56 R9700診断は完了し、通常結果との128token・MTP採否一致、HIP-only、cleanup0、元serviceのhash/UUID/healthz/readyz/clients確認と復元をPASSした。profile時23.1318 tok/sを通常性能へ流用しない。解析残件は次Phaseの参考資料とする。
- 最終sourceに対応する公開API/lifecycle、MTP有無の比較記録、必要なhost/CI、統合・公開review、履歴とplanの整理、commit/push後CI確認を残す。未測定BF16 full-model品質同等性は引き続き未証明として明記し、生成一致を品質証明にしない。

実行計画: [Phase83.5](../../../../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。
