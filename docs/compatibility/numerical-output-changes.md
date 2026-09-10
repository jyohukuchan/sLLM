# 数値・出力影響変更台帳

> 正本状態: active
> 適用開始: 2026-08-18
> 決定者: ユーザー明示指示

この文書は、同じmodel/input/sampling条件でもlogit、token列、visible outputへ影響しうる実装変更を一か所で追跡する正本である。
個別Phaseのplan/historyは詳細証拠を保持し、この台帳は変更理由、数値方向、観測された出力差、承認区分、rollback identityを索引する。

## Phase83.5最終確認

2026-09-10ユーザー指示で旧速度目標を参考値へ変更した。数値・sampling・stateの正しさは緩和しない。
R56後のCI修正はlint/formatのみで、前後source対応とfresh buildを確認する。最終比較・API・未証明の品質範囲は
[Phase83.5履歴](../history/2026/09/1-10/phase83-5-llama-guided-performance.md)に集約する。

## 数値変更の承認規則

token完全一致は観測項目として残すが、それ単独を数値correctnessのhard gateにしない。candidateは次の区分で扱う。

### N0: 数値・token互換

- 実数式、浮動小数点演算順、丸めstageを維持するか、固定matrixで必要なtoken/logit一致を確認した変更。
- 通常のcorrectness、性能、resource、fallback、cleanup条件を満たせば通常承認できる。

### N1: 解析的に誤差非増加または低減

次をすべて満たす変更は、変更前とtoken列が異なっても**数値変更として自動承認**する。高精度providerや全modelのFP64比較を
新しい必須gateにしない。

1. real-number semantic equationを変更しない。
2. dtype、丸めstage、入力集合、加算項等の欠落がなく、差の原因を演算順・近似式・精度昇格等へ局所化して説明できる。
3. 標準的な浮動小数点誤差解析により、対象演算のworst-case boundまたは期待誤差が非増加となる。例として、同符号128項の
   FP32逐次和をbalanced pairwise/tree和へ変える場合、依存深さは`127`から概ね`ceil(log2(128))`へ減る。
4. race、未定義動作、非決定atomic、未初期化値、silent fallbackを誤差低減として扱わない。同一providerのrepeatは再現可能である。
5. 既存のfinite、tiny numerical oracle、state publication、padding、cleanup、unsupported inputのfail-closedを満たす。
6. token/logit差を隠さず、この台帳へ最初の分岐位置、対象scope、source/provider identity、rollbackを記録する。

N1の自動承認は数値互換性gateだけに適用する。性能採用条件、security/correctness defect、resource、ABI、fallback、cleanup条件を
免除しない。semantic equation自体、量子化recipe、sampling規則、stop/usageを変える変更はN1に分類しない。

### N2: 誤差が僅かに増加

- 既存oracle tolerance内だが、解析上の誤差bound、accumulator精度、近似誤差のいずれかが僅かに悪化する変更。
- 自動承認しない。scope、速度・memory効果、誤差bound、token/logit差、品質controlを提示し、人間が採否を決定する。
- 「僅か」は既存の演算別tolerance内かつfinite/state/tokenization contractを壊さない範囲に限定する。範囲を説明できない場合はN3とする。

### N3: 不明・非有界・意味変更

- 差の原因が説明できない、誤差方向を分類できない、非決定、入力依存で非有界、またはsemantic equationを意図せず変える変更。
- 採用せずreplanする。人間承認だけでcorrectness/security defectを相殺しない。
- 高精度referenceはN1の定常gateではないが、N2/N3の分類を解消するため人間が要求した場合や、解析が曖昧な場合に限定して作成できる。

## 台帳に必須の項目

- 日付、Phase/変更ID、対象model/op/dtype/target/scope。
- baseline/candidateの式、演算順、accumulator、丸めstage、provider/source identity。
- N0/N1/N2/N3分類と解析根拠。
- tiny oracle、state/fallback/cleanup、同一provider repeatの結果。
- token/logit差の有無、最初の分岐位置、品質control。未実施項目は未実施と明記する。
- 性能・resource結果、採否、target split、rollback identity。

## 変更履歴

### OUT-2026-09-10-P835-ONEWAVE: small-Mのone-wave Kahan（N1候補・性能棄却）

- scope／change: gfx1201 M2/3/4、既存2tuple。ID89 StageK32をone-wave M16/N64へ縮小し、ID94の32lane分割FMA＋treeと比較した。finite K16項はE2M1/E4係数最大518,400でFP32 exact、Kahan候補は標準集約boundを増やさない方向だが、pointwise改善／出力完全一致の一般保証ではない。
- evidence: 通常6caseの全出力独立long-double oracle最大1 BF16 ULP、repeat／finite、NaN2fixtureの分類PASS。全finite scale fixtureは診断値で最大1 ULP、候補間2出力差・最大1 ULP。通常fixtureの候補間bit差は0だったが、全入力やBF16 full-model品質へ一般化しない。
- 採否: pooled40で8.974〜29.235倍遅いため棄却。one-wave StageK32は本体へ統合せず、この案の追加調整／再測定を止める。根拠は`r28-r9700-onewave-r1/`、binary SHA `aae7638a50f9de9a114ab9c68371361c3a1f4472f9d8e69292322d2c7edf0483`。詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-10-P835-FP8-PAIR-MULTIROW: FP8 qkv/zの複数行入力量子化共有（N0・公開経路統合中）

- scope／change: 既存のFP8 GDN qkv/z pairを正のM>1へ広げる。BF16入力、E4M3FNの外側scale付き重み、K5120、N10240/6144、既存target／artifact条件を維持する。b/aのBF16投影やMTP companion graphは対象外。
- 数値方針: 2つの投影が同じ入力へ別々に実行していた同一quantizerを1回にし、同じbytes／row scaleを両memberへ渡す。memberのprovider／積和／scale処理／投入順は変えない。新たな量子化やdtype変更を加えない。
- 検証状態: coreの対象predicate/routing testとHIP Rust checkはPASS。公開GPU testをM1/2/3/4/5/65/2048へ拡張し、符号／大きさの異なる行・入力更新・全出力解析oracle／独立matmul比較／repeat／provider／解放を確認する準備をした。両targetのnative build・host／公開GPU testはPASS。R31の初回通常生成ではprefill分192回のみ減り、graph容量と動的Mの比較誤りをR31bで修正した。通常MTP検証でもV2,592／R2,496回の追加削減、従来出力hash・HIP-only／cleanup0を確認した。正式速度目標とBF16品質同等性はこの結果だけで認定しない。
- 詳細: [Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。入力量子化の重複除去はllama.cpp参照方針に沿って既存sLLM pairを拡張するもので、外部source importは行っていない。

### OUT-2026-09-10-P835-ALIGNED-ID89: 整列したID89の境界判定削減（N0・gfx1201統合中）

- scope／change: gfx1201、M>=256かつ128整列、K/N=5120/17408または逆tuple。full interiorに限りK/Nを静的化し、常に真のrow／column／K境界判定を除く。M257等は現行generic lookahead、M255以下は従来の選択を維持する。K16 contribution／Kahan順序・scale・epilogueは同じ。
- evidence: scratch8 alignedケース（通常M256/M1024、M2048全finite scale、M256 NaN各2tuple）で全出力finite pair bit差0、repeat・nonfinite分類PASS。独立long-double oracleは各64点で最大0 ULP。追加M257各tupleはcontrol fallbackだけを実行し、aligned symbolを呼ばない。これはfull-model品質比較ではない。
- 採否: 通常4ケース・AB/BA両順で所要時間約1.20〜1.60%短縮、M2048も約1.22〜1.41%短縮。既定ID89の該当shapeへ統合する。新しいprovider ID／env／dtypeは加えない。通常APIとfull-model確認は未完了。
- identity: harness source SHA `56dd2527211e3c17166400239ffd00f63ea6e658130a049558c07a14287f1fb6`、binary SHA `58dfebfe4b1c409da32287a8adddff21dbfdb91c4ed0f0a7b210279e156148f3`、kernel object SHA `a6f6d614c91bbfcb107a1a26ebcda52033e7dc310df12549d459a4ace045ba35`。証拠は`r30-r9700-aligned-r1/summary.json`、[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。R9700通常serviceは同じhash・healthz/readyz200・接続0へ復元済み。

- 統合検証: r30公開経路は長すぎるkernel名のmetadata照合で失敗。r30bでNUL込み53 byteへ短縮し64 byte assertを修正した。演算変更なし。旧名archiveはscratchとISA一致だが、r30b公開6caseはoracle最大0 ULP／repeat／metadata／解放PASS、通常8192/128もHIP-only／出力hash維持／cleanup0。prefill446.442／decode23.561 tok/sの単回で、正式目標は未達。

### OUT-2026-09-10-P835-WAVE-CAUSAL: wave共通causal判定（N0・両GPU統合中）

- scope／change: gfx1030／gfx1201 MXFP8 E4のGQA6 Q8/W16 prefill。同一waveの3itemが同じquery rowを使うことから、row／valid／causal limitをwaveごとに一度計算する。無効rowをguardし、unsigned演算の意味とQK／softmax／Vの全演算・key順を維持する。
- 分類／evidence: **N0**。両GPUの通常4ケース＋NaN K/V＋large finiteの7ケースで、全headの全finite pair bit／repeat／nonfinite分類をPASS。通常2ケースはfull独立oracle、残りは各3,072点oracleを確認した。prefix1025/M129のpartial tileを含む。r25 V-only変更と合わせたr26 full-model単回では、各GPUの128token hashがr19と一致した。BF16比の品質同等性は未評価。
- 採否: 両targetの全4通常case・両順で改善し、既存kernelへsource統合する。V620 pooled時間約1.44〜2.44%短縮、R9700 prefix7168/M1024は75.000→61.362 ms（約18.18%短縮）。短いcaseへ同じ倍率を主張しない。VGPR106→102／99→95、R9700 occupancy12→16、spill0。新provider／env／dtypeは追加しない。
- identity: 修正harness SHA `8115bce25b43cf328438fc9185d9d8fe39732b8963085e132e2e8931c581afb7`、V binary SHA `025b9b74e55c10e72c77df731fda7d4ee253b5d111fd4a42628110423250dadc`、R binary SHA `29b62c4904245a1c59d55286031f4097a562645e78c057b853d3f9ead0614a19`。証拠は`r26-v620a-causal-hoist-r1/`、`r26-r9700-causal-hoist-r1/`。r1 harnessに残ったN1比較条件と上側中央値は実行前に別名r2へ訂正した。詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-10-P835-QUARTER-SCALE: ID94 quarter factor共有（N0・gfx1030統合中）

- scope／change: NVFP4 W4A4 small-M、M2/3/4・K5120/N17408と逆tuple。整数dotのquarter factorをactivation scaleへ一度掛け、4列で再利用する。dot4／FMA／wave reduction／tensor scale／BF16丸めを維持する。
- 分類: **N0**。整数dot最大2304とE4 mantissa最大15の積は16bitに収まり、両括弧順がFP32でexact。全254finite scale×4,609整数dotのhost bit一致、NaN分類一致を確認した。
- GPU evidence: gfx1030／gfx1201で通常6ケース＋全finite scale＋NaN2ケースの全finite pair bit一致、repeat／分類をPASS。通常とNaNの独立long-double oracleは最大1 BF16 ULP。全finiteの独立oracleは診断値で最大1 ULP。両GPU統合版r26のfull-model単回では各GPUの128token hashがr19と一致したが、BF16比の品質同等性は未評価。
- 採否: V620の全6ケース・両順で改善、pooled40の所要時間約0.80〜1.88%減。gfx1030 ID94だけへ既定source統合し、新しいprovider／envは追加しない。R9700は0.9953〜1.0034倍で差が一貫しないため現行を維持する。fresh r26のfull-model単回と両targetの公開ID94全6ケースはPASS。正式性能・通常API lifecycleの最新候補確認は残る。
- identity: scratch candidate SHA `5bad1e2d89cdef8407df307191237fd3df1e81c94cd409203660f9030a478569`、V binary SHA `b912654053acb104448462c459c637b4f8532a5076d081e91a457e0ae19e6971`、R binary SHA `d39299e76115d8b8cc29753e96f5f9ed530b2013e497f094a643aae78fc4628c`。V統合後の生成命令は測定済み候補と一致。証拠は`r25-v620a-quarter-fold-r1/`、`r25-r9700-quarter-fold-r1/`、`r25-smallm-quarter-integration/`。詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-10-P835-EXACT-GROUP: NVFP4の厳密4項group（N1候補・性能棄却）

- scope／change: gfx1201のscratch N32 lookaheadでfinite／非zero scaleの指数range和が3以下の場合、4つのK16 termをexactなFP32和へまとめてKahanを1回行う。条件外は従来の4回更新。比較対象は現在のN64。WMMAとscale、tensor scale、最終丸めは同じで本体未導入。
- 分類: **N1候補**。E2M1 K16とE4M3 scaleの整数係数は最大518,400、4項の絶対値和は指数差3以下で16,588,800となり24bitに収まる。groupの絶対値和が元のtermの絶対値和を超えないため、同じ項数上限での標準Kahan誤差boundを比較する。pointwise改善／token一致の保証ではない。
- evidence: R9700の11ケースで両者repeat／全出力finite・nonfinite分類、各64点の独立oracleをPASS（最大0 ULP）。全finite境界M2048のpair差はwide481出力・最大128 ULP、down239出力・最大41 ULP。これら差分点全体のoracle評価やfull-model比較は未実施。
- 採否: 全case・両順で約2.4〜2.6倍遅く性能棄却。100%group適用のfixtureでも遅いのでこの実装の追加調整は止める。VGPR180、8 waves/SIMD、LDS25,600 B、spill0という資源量だけで採用しない。
- identity: include SHA `4662aac7963cb80f5811e12a5339ce7cf2319fe14aa1f0eeadbb481cd49c5af9`、binary SHA `b635ac08445652d2f8fa74b42f9a7cd2039fb9160984b375981bc56b3202c1f1`、証拠は`r24-r9700-exact-group-r1/`。raw20配列は小数3桁・閉じ括弧欠落を記録しparser側で扱う。詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。本体未導入なのでproduction rollback不要。

### OUT-2026-09-10-P835-N32-LOOKAHEAD: NVFP4出力tile縮小（N0・性能棄却）

- scope／change: gfx1201の現在のID89 StageK64 lookaheadをM128/N64からM128/N32へ分割するscratch。各出力のK16 WMMA順、scale、Kahan、最終丸めは同じで、weight prefetchとscale staging境界だけをN32に合わせる。productionへ導入していない。
- 分類／evidence: **N0**。M127/129/257/1024/2048・wide/downとNaN scaleを含む全11ケースで、全出力pair一致、両者repeat、finite／nonfinite分類、各64点の独立oracleをPASS。最大oracle差0 ULP。full-model比較は未実施。
- 採否: VGPR228→163、occupancy6→9 waves/SIMD、LDS30,720→25,600 B、spill0だが、全shape・AB/BA両順で約6〜23%遅く性能棄却。harnessは20回の上側中央値のみを保存したため通常の偶数中央値は復元できない。raw欠落を明記し、同一binaryを再測定して結果を選び直さない。
- identity: include SHA `45e925c164495fca66a2b578ff00f5d7844cb33b8459fbe06cc6c8c028358b4a`、kernel object SHA `812f55ba1d281ee4784790d98b8fbc8459d24c2c41434eeb6f2b2ea044e85b23`、corrected harness binary SHA `905a6a166469f912c40a564e20a2056c7961bda42fcb6a1b3bfde8c80e0dd0b0`。証拠は`r23-r9700-lookahead-n32-r1/`、詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。本体未導入なのでproduction rollback不要。

### OUT-2026-09-10-P835-BLOCK16: key順block softmax（N1候補・両GPUで性能棄却）

- scope／change: 既存MXFP8 E4 Q8/W16 prefillのQK・decode・causal maskを保ち、16keyの最大値で既存stateをrescaleし、key順のdenominator加算とVのFMAを行うscratch。初回blockと例外・overflowの恐れがあるblock以降は従来式を使う。productionへ導入していない。
- 分類: **N1候補**。同じ最終weightのexp引数差は旧／blockとも最大値との差へtelescopingし、exp因子と積・和の丸め回数は増えない。signed Vは絶対誤差、underflowは共通の加法誤差を含む一様なworst-case boundで扱う。初回の従来式は共通prefixとして比較する。各入力でのpointwise改善やBF16 full-model品質同等性は主張しない。解析は`r21-block-softmax-bound.md`。
- V620 evidence: 4通常caseの全出力finite／repeat、2caseの全独立oracle（最大1 BF16 ULP）、長い2caseの各3,072点oracle（最大0 ULP）、active key15/16/17・31/32/33の候補単独oracle／repeat、NaN K/Vの分類、有限値overflow guard fixtureをPASS。通常caseのcontrolとの差は22／12／6／80出力、最大1 ULP。full-model token比較は未実施。
- 採否: AB／BA各20組の全4caseでV620は約2.70〜2.84倍、R9700は約3.0〜5.3倍遅く、**両targetで性能棄却**。raw配列から偶数個の中央2値を平均する中央値を再計算し、harnessの上側中央値とは区別する。採否は変わらず、同一binaryの再測定は行わない。R9700も同じ検査matrixをexit0で完了した。R証拠は`r21-r9700-guarded-r1/`で、元serviceとcandidate identity不変、復旧後healthz／readyz200を確認した。
- r22では有限・非正のexp引数確認を残してguard内の重複expだけを除いた。数値契約・state更新は同じで、両GPUの同matrixをPASSしたがV620約2.56〜2.70倍、R9700約2.95〜5.17倍遅く棄却。このblock16方式の追加調整を止める。source SHA `634698009caac0c4c00430b72f3f2ecf8389a5d829d0b2b25dc2db04bca24ded`、証拠は`r22-v620a-no-duplicate-exp-r1/`と`r22-r9700-no-duplicate-exp-r1/`。full-model未実施。
- identity: candidate source SHA `526cd2b3e8e2f2e43ec7873c4f3980e0569d8c1e112bf75c2d613fd368f6e34c`、control SHA `fd770f8a6bdf413e0a278a413c40f03f194f738c01f8e506b3e5815c3a20038c`、V binary SHA `3561b1ba448c426666519131c2012303cceb97eeaf903a07ce244eee7d8103af`。証拠は`r21-v620a-guarded-r1/summary.json`、詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。本体未導入なのでproduction rollback不要。

### OUT-2026-09-10-P835-ONE-EXP: online softmaxのexp(0)除去（N0候補・性能棄却）

- scope／change: MXFP8 Q8/W16 prefillだけで、finite scoreとfiniteまたは初期-INFのmaximumについて、必ず1となるexp(0)を省略する。nonfinite経路は元の2-exp式を維持し、QK／denominator／V積算／BF16丸めは変更しない。本体への導入はない。
- evidence: 両GPUの数値式10caseで各更新段階のfinite bit／nonfinite分類とhost oracleをPASS。attention4caseの全出力control bit／repeat／finite、うち2caseの独立oracleをPASS。長い2caseまで独立oracleを実行したとはしない。R9700の最初のhost harnessはtarget定義違いで起動前にexit2となり、別binaryで訂正した証拠だけを使う。
- 採否: V620は全4caseで0.38〜1.33%遅い。R9700も主要caseで約0.4〜0.5%遅く、M129の約0.15%差は両順序で再現しないため**両targetで性能棄却**。資源量は不変でも全体の演算費用削減にはつながらず、full-modelへ進めない。
- identity: `.local-artifacts/phase83-5/r20-one-exp/v620a-r1/`と`r9700-r2/`。失敗したR9700 `r9700-r1/`も保持する。詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-10-P835-FIXED-ACCUM: NVFP4整数累積scratch（N1候補・性能棄却）

- scope: gfx1201 ID89の現行StageK64 lookahead、K5120／17408。productionは変更していない。
- change: E2M1 K16 dotとfinite E4 scaleを整数のQ20表現へ移し、外側のFP32 Kahan累積をsigned64のexact和へ置換する。K17408までの絶対値上限はsigned64以内。最終FP32変換、tensor scale乗算、BF16丸めを維持し、NaN scaleは明示的にnonfinite出力へ分岐する。
- 分類: **N1候補**。finite scopeの実数式を保持し、外側累積をexactにする誤差解析を記録した。旧Kahan出力との完全一致は採用条件にしない。GPU試験は独立long-double oracleを各64点、repeatと全出力のfinite／nonfinite分類で確認した限定証拠である。
- output: M2048の全finite scale境界fixtureではcontrolとの差が975出力、最大47 BF16 ULP。sampled oracleは両者最大0 ULPだが、差分975点全体の独立精度検証ではない。NaN fixtureは両者5,376 nonfinite、強制sample3点も分類一致。full-model token／BF16品質比較は未実施。
- 採否: AB／BA各20組の全6caseで約1.71〜1.85倍遅く**性能棄却**。追加integer-WMMA派生は行わない。SGPR spill53はVGPR laneへ退避しprivate segment0だが、命令数は増加した。spillのみを遅延の唯一の原因とは断定しない。
- 後続FP64候補（r18）: 同じFP32 termをFP64でK昇順に加算し、最後にFP32へ丸める。絶対誤差boundは`[u32+(1+u32)gamma_1088(u64)]Σ|term|`で、標準的なFP32 Kahanの約`2u32Σ|term|`より小さいN1候補。各64点oracle／repeat／nonfinite分類はPASSしたが約4.8〜4.95倍遅く棄却。finite境界975差分／最大47 ULP、full-model未実施。証拠は`fp64-outer-r18/`と`r18-r9700-candidates-r1/`。
- identity: 現行control section SHA `3c28200e573acfccc5e52ae5a0ed28ed6d5c5fa65f1d52b95d72467a4d8ae754`。scratch `fixedpoint-r17/`、GPU証拠`r17-r9700-fixedpoint-r1/`。本体への導入がないためproduction rollbackは不要。
- 詳細: [Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。


### OUT-2026-09-10-P835-PREFILL-PAIR: NVFP4 gate/up量子化共有（N0・既定統合済み）

- scope: exact gfx1030／gfx1201、既存NVFP4 gate/up projection pair、M>=64、K5120／N17408。M1の既存経路と対象外の分解は維持する。
- change／分類: **N0**。同一入力・同一scaleの量子化を1回へ共有し、各projectionのprovider・演算順・丸めは変えない。queue scratchの排他は量子化から両projectionのenqueueまで保持する。
- correctness: 初回r16はscale planeのoffsetにM乗算が欠けてGPU FAIL。r16bでchecked layoutをsizing／execute間で共有して修正。両GPUのM64,65,127,128,129,512で独立2matmul比較・数値oracle・repeat・量子化1回・cleanupをPASSした。
- output: 同一r16b binary、8192/128、T1/P.95/K20、seed123、MXFP8 KV、MTP幅2のoff/onで、各GPUの128token hashが一致（分岐なし）。V620 MTP74/107、R9700 76/104も不変。これは1入力の比較でありBF16 full-model品質同等性の証拠ではない。
- 性能／resource: 重複量子化224回を削減し、workspace／request high-waterは不変、HIP-only／cleanup0。単回prefillはV62039,997.216→40,537.306 ms、R970019,081.115→19,146.437 msで改善を確認できず既定採用を見送る。その後、1 fenceに揃えた実chunkの同一process AB／BA各20組で中央値約1.0〜1.8%の費用削減を確認した（V620 M2048は順序差あり）。全体改善の認定とは分け、既定採用時のrollback互換性を確認中。
- identity／rollback: r16bの実測は`candidate-gfx1030-r16b-prefill-shared`／`candidate-gfx1201-r16b-prefill-shared`のsource・binary digestへ対応し、そのbinaryでは明示1だけ有効。r19のsourceでは無指定／1を有効、0／不正値を従来分解へ変更した。native providerがprepare段階でUnsupportedを返した場合も従来2matmulへ戻す。その他のprepare失敗・submit後の失敗では戻さない。focused host検査と下記のr19公開GPU／full-model検証を通過した。
- r19確認: 両targetの8192/128単回で共有env未指定の既定経路、r16b明示onとの128token一致、HIP-only／cleanup0を確認。通常APIの短文／SSE／cancel／recovery／8192/128も両者PASS。これはPhase83.5速度目標の正式達成やBF16品質同等性の認定ではない。
- 詳細: [Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。証拠は`v620a-r16b-prefill-shared-comparison.json`と`r16b-r9700-prefill-shared-r1/batch-summary.json`。

### OUT-2026-09-10-P835-M3-DOWN-LDS: ID94 activation共有（N0・既定統合済み）

- scope: exact gfx1201、NVFP4 W4A4、M3、K17408／N5120のDown projection。gfx1030、M2/M4、wide M3は既存bodyを維持する。
- change／分類: **N0**。3行のdecoded activationとscaleを64個のK16 blockごとにLDSへ共有する。weight prefetch、laneごとのK順、dot4／FMA／shuffle reduction、tensor scaleとBF16丸めは保持し、消費前・上書き前に同期する。既存LUTとID94を使い、新しい公開symbolや選択envは追加しない。
- evidence: scratchの全15,360出力で独立host oracle最大0 ULP、control bit一致／repeat／finiteを確認。BAの時間格納ラベルを凍結sourceとraw配列から補正した比較はcontrol0.118519／candidate0.1137185 ms、AB／BAとも約4%短縮。wide側は約0.27%で採用しない。証拠は`r18-r9700-candidates-r1/batch-summary-corrected.json`で、補正前のpool中央値を使わない。
- integration: r19 sourceへDownだけ統合。gfx1201 compileは93 VGPR／49 SGPR／LDS4,896 B、spill0。公開GPUのM2/3/4×2tupleは両targetで全出力oracle最大0 ULP／repeat／finite／解放をPASS。R9700 full-model単回の128tokenはr16bと一致、通常API5caseもPASS。prefill416.296／decode23.456 tok/sで全体速度目標は未達であり、scratchの局所短縮と区別する。切戻しには既存ID94 provider制御またはsource identityを用いる。
- 詳細: [Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。


### OUT-2026-09-10-P835-QTILE8-B4: gfx1030のK/V 4行staging（N0）

- scope: 既存gfx1030 MXFP8 Q8/W16 prefill provider。selector、query/head geometry、encoding範囲は変更しない。
- change: K/Vを4行ずつLDSへ読み込み、key昇順、QK reduction、owner-lane online softmax、V積算、BF16丸めを維持する。
- 分類: **N0**。同一演算順のままloadと同期をまとめる。gfx1201は実測で悪化したため元の1行stagingを維持する。
- evidence: scratchの4caseで全出力control bit一致、repeat／finite、独立oracle最大1 ULP。r15 production公開経路の13境界caseもoracle最大1 ULP、Q4 controlとの全出力一致、dispatch／cleanup PASS。V620 full-model 8192/128のM3有効runはr13の128token hashと一致したが、単独変更の速度効果は未確定。
- 性能: V620 scratchで約0.65〜0.80%改善、R9700で約1.7〜4.0%悪化。r15既定構成の正式1 warmup＋3 measuredはprefill中央値198.577009079／decode23.795903167 tok/sで、prefill200の目標は未達。scratch改善をfull-modelの速度達成へ読み替えない。
- rollback: gfx1030を同じkernel内の既存1行staging bodyへ戻す。
- 詳細: [Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。raw証拠は`.local-artifacts/phase83-5/qtile-kv4-r14/`と`r15-public-mxfp8-prefill-gfx1030/`。


### OUT-2026-09-10-P835-QUEUE-SCRATCH: 低精度matmul作業領域のqueue再利用

- scope: M>1のNVFP4／MXFP4／MXFP8／MXFP6 matmul。M1とFP8 outerの個別確保は維持する。
- change／分類: **N0**。planごとの一時領域を同一queueの再利用領域へ移す。量子化・演算・丸め順とproviderは変更せず、同一streamへの量子化とmatmulのenqueue全体をmutexで保護する。異なるqueueは別領域を持つ。
- correctness: r12の公開NVFP4 small-Mは両GPU・両tuple・M2/3/4の全出力oracle最大0 ULP、prefillは非整列境界でsampled oracle／全出力finite・repeat／cleanupをPASS。r13ではMXFP4/8/6の既存M>1を含むsynthetic／実重みmatrixも両GPUでPASS（最大relative error約0.00390、HIP-only、cleanup0）。
- resource: 通常APIの8192/128・context8320・自動chunk2048は両GPUでHTTP/SSE200、HIP-only、cleanup0。r11で再現したOOMを解消した。単回の完走であり速度目標達成の判定ではない。
- output: r11の同条件APIはOOMのため完走token比較がない。生成例をBF16 full-model品質同等性の証拠とはしない。
- identity／rollback: gfx1030 `r12-queue-scratch`、gfx1201 `candidate-gfx1201-r12-queue-scratch`の固定source／binary digestと`r12-public-*`証拠。rollbackはr11のplan個別確保で、通常APIのOOMも再発し得る。
- r13はprepareとallocation-free footprint queryのchecked layout計算を共有し、作業領域サイズをadmissionへ計上する変更。演算式は不変。ホストquery回帰、両GPU公開matmul、通常API8192/128をPASSし、同条件r12 APIと生成textが一致した。
- 詳細: [Phase83.5実行計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-10-P835-M3-CHECKPOINT: MTP部分採用時のGDN prefix保存

- scope: exact gfx1030／gfx1201、M3検証のrow0／row1、対応linear-state backend。Qwen3.8 27B NVFP4／MXFP8 E4 KV／固定GPU sampling／MTP幅2で実モデル確認。
- change: M3中のconv／FP32 recurrent stateを2 planeへ保存し、部分採用時は選択planeをactive slotへコピーする。pre-block slotは保持し、KV lengthをtrimする。全targetのM1／M2 replayを省略するがsampling・RNG・companion stateの規則は変更しない。
- 分類: **N0（M3 state保存と下記固定matrixのtoken互換scope）**。保存自体はM3の演算順・丸めを変更しない。legacy replayはM1／M2 matmulを使い丸めが異なり得るため、任意入力のbit一致へ一般化しない。
- 数値: 新private runtime GPU testはqk16／value48／d128／conv4、非zero初期state、accept1／2／full3、次M1、再arm、stale／invalid row、巻戻しを両targetでPASS。独立oracleの最大BF16差0／3 ULP、conv bit一致、FP32 recurrent最大絶対差約5.30e-7／5.46e-7。テストoracleのK offsetとgfx1030物理layoutの誤りを修正し、失敗logも保持した。
- full-model: r9b→r10は両GPUとも同一8192/128 fixture・seed123・T1/P.95/K20で128token hash一致（初回分岐なし）。各1測定であり品質集合の検証ではない。BF16 full-model品質同等性は未証明。
- 性能／資源: decodeはV620 19.335→24.176、R9700 18.479→22.893 tok/s。request追加293.625 MiB、allocation peak25,021,492,176 B、cleanup zero。正式反復と公開lifecycle確認は未完了。
- r11の一括commitは同じplaneのcopyをまとめ、最後の1 fence後だけ全stateを公開する。3stateのGPU payload／次M1、preflight無変更、native hostのcopy／fence失敗poisonを確認。両GPUの同一8192/128単回tokenはr10と一致し、N0の同じscopeを維持する。新しいmodel品質同等性の主張はない。証拠は`r11-public-checkpoint-*`／`r11-comparison.json`。
- identity／rollback: `candidate-gfx1030-r10-mtp-checkpoint`／`candidate-gfx1201-r10-mtp-checkpoint`のsource・binary digest、GPU `r10-public-checkpoint-*-r3`、full-model `r10-comparison.json`。戻す場合はr9bのlegacy target replayへ戻し、capture／lazy allocationも無効化する。未対応backendは実行前Unsupportedでlegacy replayを使う。
- 詳細: [Phase83.5実行計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-09-P835-QTILE8-OWNER-SOFTMAX: query別owner laneのsoftmax（N0）

- scope: exact gfx1030／gfx1201、既存QTile8/W16 MXFP8 E4、q24／kv4／d256、M>=128、prefix>=1024。
- baseline/candidate: wave内3queryのQK順を維持し、lane0で順番に行ったsoftmax更新をowner lane0/1/2へ分配する。
  ownerごとにmaximum／denominatorを保持し、各queryのrescale／contributionだけをbroadcastする。
  key順、FP32式、V累積順とBF16丸めは維持する。kernel symbol／selector／KV配置は変更しない。
- 分類: 両targetの固定fixtureで全出力control bit一致を確認した範囲の**N0**。
  境界の独立oracleと長caseのsampled oracleは実行範囲を区別し、full-model BF16品質の証拠とはしない。
- correctness: prefix1024/M128、1025/M129、8193/M128、7168/M1024の4caseで両targetとも
  fullcontrol／repeat bit一致、nonfinite0、exit0。境界oracle最大1 ULP、7168/M1024の独立oracleは
  3query×2heads×全256次元のみで最大0 ULP。初期large-range fixtureで既存control自身に9 ULP／abs0.5差が
  出た診断は未解決として保持する。許容差を変更してPASSにしていない。
  r9b公開GPUは両targetでQTile4/QTile8切替境界を含む既存matrix、独立oracle最大1 ULP、
  QTile4比較0差、repeat／cleanupをPASS。r9のQTile4移植ミスは公開検査で検出し修正済み。
- 性能: 200ms warmup＋20 AB／BAで7168/M1024はV620 139.922→135.944ms、
  R9700 78.781→74.657ms。両targetともprivate segment0、LDS2048、occupancy維持。
  単体の約2.8%／5.2%短縮をモデル速度へ換算しない。
- 不採用案: 最初のscore配列版はV620の8193/M128で約8.5%遅く、private segment16 B/laneのため棄却した。
  初回divergent shuffle誤りは修正したが、修正後の測定でも遅かった。全診断は`qtile8-softmax-ilp/`に保持する。
- rollback: QTile8 bodyをowner-lane変更前へ戻す。既存の明示QTILE4／baseline選択も維持する。
- 詳細: [Phase83.5実行計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-09-P835-GFX1201-FP8-SMALLM: hipBLASLt algorithm identity選択（N0）

- scope: exact gfx1201、E4M3FN outer-scale FP8、M2..4、hipBLASLt version100401／revision cd957402。
  対象K/Nは5120/6144、5120/10240、5120/12288、5120/1024、5120/17408、6144/5120、17408/5120。
- baseline/candidate: 既存rank0から、同一descriptorのsuccessful／zero-workspace heuristic候補に含まれる
  index123373（K5120/N6144、K5120/N1024、K6144/N5120）、123374（K5120/N10240、K5120/N12288、
  K17408/N5120）、123375（K5120/N17408）を選ぶ。配列順位で選ばず、未知library identity・候補欠落時はrank0を維持。
  M1／M17と明示override、FP8量子化、scale、BF16出力契約は変更しない。
- 分類: 固定matrixの全出力比較で数値一致を確認した範囲の**N0**。library内の全入力に対する演算順一致は未証明であり、
  kernel名や単一生成例だけでBF16参照品質同等性を主張しない。
- correctness: 2tuple×M2/3/4×10候補の全60候補で全出力rank0比較・独立FP32 FMA→BF16 oracle最大0 ULP、
  finite、3回repeat一致。公開runtimeのr7 controlも2tuple×M1..5で最大0 ULP／cleanup PASS。
  r8公開runtimeも同10ケースで全出力oracle最大0 ULP／repeat／cleanup PASS。rocprofでM1..4の
  MT16x16x32とM5の旧MT16x128x32を確認した。r8 full-model単回も前候補と128出力hash一致、cleanup PASS。
- 性能: 200ms連続warmup＋20 AB／BAでrank0約90〜132µsに対し選択候補約33〜46µs。
  約2.7〜2.85倍は単体の観測値で、モデル改善率ではない。
- 追加5tuple: M1..4×10候補の200行で全出力rank0比較／独立oracle最大0 ULP、finite／3 repeat一致。
  初回wrapper終了137と途中継続の意図的停止130を単独PASSとは扱わず、完了済み行と最終継続exit0を結合した証拠。
  M2..4の単体倍率は約1.29〜2.96倍で、M1は既に調整済みのため新しい改善とは数えない。
  r8公開controlとr9追加policyの両方が7tuple×M1..5で全出力oracle最大0 ULP／cleanup PASS。
  r9 traceで追加index123375相当のMT16x16/WSGRB1を9dispatch確認した。r9bとの差はattention sourceだけで、
  FP8証拠の対応を`r9b-fp8-evidence-mapping.json`へ記録した。
- rollback: r7のM2..4 single-result/rank0 policy。library identityとopaque algo選択modeをplan/cacheへ保持する。
- 詳細: [Phase83.5実行計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-09-P835-GFX1030-MXFP8-DECODE: staged32のvalue/scale decode統合（N0）

- scope: exact gfx1030、MXFP8 E4、staged32 decode stage1。Q8/W16 prefill、旧8-wave、gfx1201、E5は対象外。
- baseline/candidate: E4値とE8M0 scaleを別々にfloatへ変換して乗算する処理を、既存E4 `decode_scaled`へ統合する。
  KVロード順、QK／online softmax／V累積順、BF16丸めは維持。256×256コードのhost exhaustiveで有限値bit一致、
  NaN class一致を確認し、GPU固定matrixでも全出力一致した範囲を**N0**として扱う。
- correctness: V620-B scratchのprefix1023／1024／1025／8192／8193、M1／3でfull control／repeat bit一致、
  finite PASS、独立oracle最大1 ULP。scale 0xffの別probeは両方1536非有限値、hash／repeat一致で、伝播契約の証拠とする。
  r8公開runtimeはlength1023／1024／1025・M1..4で既定ID93切替、独立oracle最大1 ULP、
  finite／repeat／cleanup PASS。r8 full-model単回も前候補と128出力hash一致、cleanup PASS。
- 性能: 200ms warmup＋20 AB／BA、8192／8193 prefixではM1約17.8%、M3約29.6〜29.7%短縮。
  VGPR53→48、occupancy16維持、spill0。gfx1201は同案で遅くなったため採用しない。
- rollback: gfx1030 staged32でも旧value/scale別decodeを使うr6 attention body。
- 詳細: [Phase83.5実行計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-09-P835-ID89-LOOKAHEAD: NVFP4 StageK64先読み（N0）

- scope: exact gfx1201、NVFP4 W4A4、`(K,N)=(5120,17408)/(17408,5120)`、`M>=256`。
- baseline/candidate: rolled StageK64の単一LDSを2 parityへ分け、次のK64 tileのpacked値とscaleを
  VGPRへ先読みする。各出力の4個のK16 WMMAとKahan加算順、量子化decode、scale、BF16 RNEを維持するため**N0**。
  logical ID89を維持し、deviceは`sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_kahan_lookahead_v1`へ変わる。
- correctness: M256 wide／M257 down／M1024両tupleのscratchは全出力control／repeat一致、finite PASS、
  64点独立long-double oracle最大0 ULP。r7公開GPUのM255／256／257両tupleも境界oracle最大0 ULP、
  metadata切替／全出力finite／repeat／cleanupをPASS。r7モデル8192／128単回はHIP-only／非有限0／cleanup zero、
  128出力tokenはr6と一致した。
- 採否: 200ms warmup＋20 AB／BAで約1.16〜1.26倍のため上記scopeへ統合。
  VGPR127→228、LDS15360→30720 bytes、spill0。rollbackは旧StageK64 symbolで、StageK32も保持する。
- 詳細: [Phase83.5実行計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-09-P835-ID87-TILE128: NVFP4 prefill行tile拡大（N0）

- scope: exact gfx1030、NVFP4 W4A4、`(K,N)=(5120,17408)/(17408,5120)`、`M>=512 && M%128==0`。
- baseline/candidate: ID87のTileM64をTileM128へ拡大し、weightのロードをより多くの行で共有する。
  同じtemplate bodyで各出力のK順序、dot4、補償加算、scale、BF16 RNEを維持するため**N0**。
  logical ID87は維持し、device symbolは`sllm_nvfp4_w4a4_prefill_compensated128x64_v1`へ変わる。
- correctness: scratchの両tuple・M127／128／129／512／1024で全出力control／repeat一致、
  境界sample oracle最大0 ULP、finite PASS。r6公開GPUでは両tuple・M511／512／513／1023／1024／1025の
  device切替、境界5×5 oracle最大0 ULP、全出力finite／repeat／cleanupをPASS。r6モデル8192／128単回もPASSし、
  128出力tokenはr5と一致した。FP8 M1変更も同時に入った測定のため、モデル速度差をTileMだけへ帰属させない。
- 採否: M512／1024で約2.8〜6.3%高速化し上記scopeへ統合。M129は約21%遅いため端数行へ拡張しない。
  VGPRは103→152、LDSは6144→9216 bytes、spill0。rollbackは旧TileM64 device symbol。
- 詳細: [Phase83.5実行計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-09-P83-GFX1201-RESIDENT-KV: gfx1201通常KVのresident固定（N0）

- scope: Rust HIP adapterがexact `gfx1201`向けに作る、sliding windowなしの通常KV stateの全logical capacityと
  FP16／FP8／MXFP8／NVFP4 encoding。exact `gfx1030`のcapacity 65,536以上、exact `gfx942`の全capacityという
  既存resident選択は維持する。direct native C ABIの`CAPABILITY_SELECTED`／明示`VIRTUAL_CONTIGUOUS`、sliding state、
  unknown target、他targetは対象外である。
- baseline/candidate: baselineはgfx1201のcapacity 65,535以下で`CAPABILITY_SELECTED`を渡し、HIP VMM
  `virtual-contiguous`を選ぶ。candidateはexact targetを確認したcreate時に`CONTIGUOUS_RESIDENT`を明示する。同じopaque owner、
  token-major logical layout、K/V value／scale／outer-scale plane、append／attention kernel、logical pointer arithmeticを使い、
  provider error後のretryや別encodingへのfallbackは行わない。
- 分類: **N0**。変更するのは物理allocation ownership、commit accounting、fork時の物理copy方式だけであり、入力集合、KV byte
  encoding、実数式、浮動小数点演算順、丸めstage、state publication順を変えない。contiguous forkはVMMのread-only page共有／
  tail COWではなくsame-device D2D cloneを使うが、公開されるlogical bytesは同じである。旧VMM経路で壊れた出力は数値baselineではなく
  correctness failureとして扱う。diagnosticの`kv_memory_kind`とphysical committed-byte値はprovider変更を反映して変わる。
- diagnosis: canonical R9700のscratch r22では、短い通常要求、SSE、cancel、recovery後の8,192-token prefill＋MTP verify中、
  layer 51／55のVMM grow直後かつappend kernel前にlive layer 3 key／value prefix先頭128 byteがexact zeroへ変化した。
  device readbackとsyncは成功し、watch対象と新規pageのvirtual addressおよび記録したallocation handle値は異なっていた。
  これにより差をHIP VMM live-mapping境界へ局所化したが、ROCm内部の根本原因は未確定である。
- correctness/resource: resident選択を入れたscratch r23は同じ履歴から8,192入力／128出力を完走し、layer 3のMXFP8
  key/valueと両scale planeがMTP verify前後およびrewind後にbyte一致、replay attentionの非有限値0、HIP-only、fallbackなし、
  cleanup 0だった。capacity 8,320でresident KV committed bytesは`281,149,440`。CPU／GTT fallbackや常駐FP16 mirrorはなく、
  resident providerはlogical capacity全量をcreate時にdevice allocationする。VMM page sharingの物理memory効率は利用しない。
- 決定・検証状態: exact `gfx1201`の通常KVを全capacityでresidentへ固定するcorrectness変更を採用する。r22/r23はscratch
  diagnostic/candidate evidenceである。current-main sourceの両GPU target build identity確認とKV host test 28件はPASSし、final
  GPU API／CLI実行は進行中である。direct ABI／sliding VMMは診断・research経路として残す。
- 詳細: [Phase 83計画](../plans/archive/2026/09/1-10/phase83-mxfp8-fixed-sampling-mtp.md)、
  [KV memory decision](../architecture/kv-memory.md)。

### OUT-2026-09-09-P83-BF16-PREFILL64: ID91 BF16 prefill共有tile転置（N0）

- scope: exact gfx1030、BF16入力／重み、`M>=64`、`(K,N)=(10240,5120)/(5120,17408)/(17408,5120)`。
- baseline: ID2 tiled16。candidateは64×64／K32 tileで共有weightを`[K][N]`配置にし、ロードを共有する。
  各出力の昇順K、BF16からFP32への変換、FP32積和の演算順、最終BF16 RNEを維持するため**N0**。
- correctness: V620-Bのscratch 10ケースで全出力control一致、各ケース32要素の独立long-double oracle差0 ULP、repeat一致。
  当初の公開runtime M1024 probeは先頭8行／16列だけをoracleと比較しており、64行以降の未書込を見逃した。
  累積レビューで1D launcher／2D kernel indexの不一致を検出し、1D tile indexを行・列へ復号する修正を行った。
  修正版は公開plan／executeによるFC／up／down × M63／64／65／1024の12ケースで、64行以降と末尾を含む
  独立BF16 oracle、有限値、guard領域をV620-BでPASSした。M63はID2、M64以上はID91を確認した。
  証拠は`phase83/id91-public-launch-r26/identity.json`。旧r25 V620モデル実行は数値受入に使用しない。
- performance: 直接kernelを起動したscratchのM1024 FC／up／downはID2比60.7%／68.0%／67.2%短縮。
  旧公開runtimeのup 57.1144 msは未書込のある実装の値であり、正しい公開経路の性能証拠から除外する。
  修正版の公開性能・MTPを含むモデル全体の速度目標は未確認で、追加最適化はPhase83.5で扱う。
- 決定: 上記shape条件で既定選択する実装を追加。最終buildでのモデル検証・CI確認はPhase83の残作業。
- 詳細: [Phase83実行計画](../plans/archive/2026/09/1-10/phase83-mxfp8-fixed-sampling-mtp.md)。
  ローカル探索identityは`phase83/bf16-prefill64-gfx1030-r2`、公開経路identityは`phase83/bf16-prefill64-gfx1030-integration-r1`。

### OUT-2026-09-09-P83-FP8-FUSED-M2-4: ID92 FP8 outer fused M2--4（N1）

- scope: exact `gfx1030`、FP8 outer E4M3FN、`K=5120`、`M=2..4`、`N=10240`（GDN qkv）、`N=6144`（GDN z）または`N=248320`（lm_head）。`M=1`、FNUZ、別の`K/N`、別targetは既存providerへ戻す。
- Phase83.5の追加scope（公開経路検証中）: exact gfx1030、E4M3FN、`M=2..4`、`K=6144/N=5120`へ同じID92を拡張する。既存ID82 bodyのtupleを`6144/5120/12 groups`として各rowへ適用し、M1・FNUZ・範囲外shapeを変えない。K=6144では約96更新＋5段のwave reductionに対し、ID71は約3072更新であり、同じ前提のworst-case boundが増加しないN1として扱う。V620-Aのscratch probeはM2/3/4で独立sample oracle最大0 ULP、全出力のID71比較も0差だった。単体中央値は約50.6／58.1／160.2 µsだがcontrolの分散が大きく、比率を安定した高速化率とは認定しない。raw source／oracle／測定は`.local-artifacts/phase83-5/fp8-smallm-experiment/`。r2の公開runtimeではM1〜5の全出力oracleとselector／cleanupを確認し、V620の8192／128・MTP単回もPASSした。正式反復の速度とBF16 full-model品質同等性は未認定。
- Phase83.5の追加4 tuple（統合中）: `M=2..4`、`(K,N)=(5120,12288)/(5120,1024)/(5120,17408)/(17408,5120)`。同じfused bodyを10／34 groupsで再利用し、lane内dot2更新＋5段shuffleはK5120で80＋5、K17408で272＋5。ID71のK/2更新に対するN1分類を維持する。V620-Aのscratchでは12ケースすべてが全出力ID71一致、独立FP32-FMA／BF16 oracle 144点で最大0 ULPだった。単体中央値は約4.4〜10.6倍だが、公開runtimeとfull-modelの速度へ一般化せず、r4で確認する。詳細は`.local-artifacts/phase83-5/fp8-smallm-more/`。
- Phase83.5 r4公開検証: 追加5 tuple×M1〜5×3反復の75実行で全出力BF16 oracle最大0 ULP、公開selector／device symbol／cleanupを確認した。証拠は`.local-artifacts/phase83-5/fp8-smallm-public-r4/`。V620-Aの8192／128・MTP幅2単回はprefill198.308／decode16.766 tok/s。正式反復とBF16 full-model品質同等性の証明ではない。
- baseline/candidate: 通常のM>1 providerはID71 `matmul.fp8.outer.prefill.gfx1030.half2.64x64.v1`であり、64x64 tile、K32 LDS staging、FP32 accumulatorを使う。ID92は一つのCTAで2--4行を同時に計算し、GDNのN10240/N6144では既存ID82のLDS LUT E4M3FN decode、lm_headのN248320では既存ID68のdword8 E4M3FN decodeを行う。各rowの全K項、activation／weight scale、`float_to_bf16_rne_bits(acc * activation_scale * weight_scale)`を保持するが、ID71のthread-local K順とは異なり、各waveの部分和をordered shuffle treeで結合する。したがって、M1 provider（ID82/ID68）とのrowwise算術はbitwise一致する一方、通常M>1のID71との演算順はN0とは分類しない。
- 分類: **N1（上記scope内）**。K項集合、E4M3FN decode、FP32 accumulator、scale適用、BF16 RNEを変えない。ID71の一出力はK=5120で約2560個のhalf2 dot更新をthread-localに逐次蓄積するのに対し、ID92は各laneの約80更新を5段のwave shuffle treeへ渡すため、候補の依存深さは概ね`gamma_85`（80更新＋5段）で、ID71の概ね`gamma_2560`より増加しない。これは標準浮動小数点boundに基づくN1分類であり、pointwiseなbitwise一致や全shapeへの一般化は主張しない。
- 数値検証: V620-A `gfx1030`の公開runtime probe（GPU UUID `GPU-76a08c022586fed6`）で、3 exact shapeの`M=1..4`、repeat、finite、独立sampled E4M3FN/BF16 oracle、cleanupを確認した。M2--4は全shapeでID92、`dispatch_count=2`、oracle最大BF16 ULP差0、resources releasedを得た。比較用の既存ID71 probeも同じfixtureでfinite、oracle最大ULP差0、ID82/68 rowwise出力との差0 ULPだったが、これはfixture結果であり、ID71との一般的なbitwise互換性を意味しない。公開probe identityは`.local-artifacts/phase83/fp8-id92-public-gfx1030-r1/summary.json`に記録する。
- 性能・採否: 同じV620-A公開probeのM2/M3/M4中央値は、GDN qkvが`0.306523/0.355844/0.570367 ms`、GDN zが`0.229922/0.263883/0.399765 ms`、lm_headが`2.603708/2.614028/2.633307 ms`だった。これはoperator-levelのtarget／shape限定証拠であり、full-model出力、MTP性能、別targetへの採用を示さない。exact shapeでは既定selectorがID92を選択する。
- rollback: ID92のkernel/provider選択を無効化またはshape条件を外すと、gfx1030の既存ID71（M>1）へ戻す。`M=1`は既存のID82／ID68を維持し、FNUZ、非有限payload、未対応shape／targetは既存fail-closed規則に従う。
- 詳細: [Phase 83計画](../plans/archive/2026/09/1-10/phase83-mxfp8-fixed-sampling-mtp.md)、ID92 public probe summary: `.local-artifacts/phase83/fp8-id92-public-gfx1030-r1/summary.json`。

### OUT-2026-09-09-P835-FP8-M1-TUPLES: ID82 FP8 M1 tuple拡張（N1・統合中）

- scope: exact gfx1030、E4M3FN、M=1、K/N=5120/12288、5120/1024、5120/17408、6144/5120、17408/5120。M2〜4のID92、FNUZ、他targetと明示rollbackは維持する。
- candidate: 測定済み`fp8_outer_decode_gfx1030_fused_id82_body<Rows=1>`を独立M1 device wrapperで呼ぶ。M2〜4と同じwrapperに混ぜて最大VGPRを共有しない。LUTでのexact FP8→FP16変換、FP32 dot2／wave reduction、outer scaleとBF16 RNEを維持する。N1分類は既存ID82の前提を引き継ぎ、実験の全出力一致だけでN0とはしない。Kは積の個数であり依存深さではない。lane内dot2更新はK5120で80、K6144で96、K17408で272、後段wave shuffleは5段。
- evidence: V620-B scratchの5tupleは全出力ID68比較／repeat／finiteをPASSし、独立境界sample oracle最大0 ULP。200ms連続warmup＋20組AB／BAで1.679／2.501／1.514／1.597／1.693倍。初回の短いwarmupではclock過渡があったため、安定比較を採用根拠とする。VGPR170／spill0は性能判定の代替にしない。
- status: `.local-artifacts/phase83-5/fp8-m1-more/`にraw証拠を保持する。新buildでの公開GPU・モデル全体・正式反復は未完了。

### OUT-2026-09-09-P83-GQA6-QTILE4-MXFP8: GQA6 QTILE4 MXFP8 prefill（N1・Phase83.5で既定化検証中）

- scope: `q_heads=24`、`kv_heads=4`（GQA ratio 6）、`head_dim=256`、query count `>=128`、standard OCP MXFP8 E4M3 block32／E8M0 KVを対象とする。解析の入力範囲は有限で実用的な通常値に限定し、別のhead幅・GQA ratio・特殊値の一般化は主張しない。Phase83当初の性能証拠はV620 `gfx1030`だけであり、Phase83.5の両target探索は下記へ追記する。
- baseline/candidate: baselineのgeneric causal attentionは256個のQK積をFP32でbalanced reduction treeへ渡し、依存深さは8段である。candidateのQTILE4は各laneの8積を4組のpairへまとめ、pair treeと5段のwave shuffleで合計8段にする。両方とも同じMXFP8 decode／E8M0 scale、FP32 causal online maximum・denominator・weighted-V更新、key順序、BF16 RNE出力を維持する。256次元の暗黙scaleは両経路とも`1/16`である。
- 分類: **N1**。実数式、入力項、dtype、丸めstageを変えず、QK reductionの標準上界は両経路とも概ね`gamma_8 * Σ|product|`であり、candidateのworst-case boundは増加しない。これはpointwiseなBF16誤差改善やbitwise一致を保証する主張ではない。resident FP16 mirrorを必要としないMXFP8直接decodeの分類であり、常駐FP16複製を許可する変更でもない。
- 数値gate／採否: 上記のN1分類は本台帳と[Phase 83計画](../plans/archive/2026/09/1-10/phase83-mxfp8-fixed-sampling-mtp.md)の規則に従う数値互換性gateへ自動適用する。ただしproduction既定化・採用は別判断であり、target／shapeを限定したGPU correctness、performance、fallback、cleanup、API／MTP統合の確認を要する。Phase83完了時は`SLLM_CAUSAL_ATTENTION_GQA6_QTILE4=1`を明示opt-inとして保持した。
- 探索性能: V620 `gfx1030`、8,192入力／128出力、MXFP8、ID87 compensated prefill＋QTILE4、MTP無効、warmup 0／measured 1の診断値は prefill **211.0486 tok/s**、decode **7.8054 tok/s** だった。これは単回の探索値であり、Phase 83のMTP有効200／20目標、正式な反復性能比較、既定採用の証拠ではない。ユーザー目標は未達・未完了である。
- 検証状態: 独立MXFP8 prefill GPU oracleはV620 `gfx1030`とR9700 `gfx1201`で完了した（`.local-artifacts/phase83/mxfp8-prefill-gfx1030-r1`／`mxfp8-prefill-gfx1201-r1`）。query127／128／129、prefix0／31／256で最大BF16 ULP差1以内、当該fixtureのprovider間出力はbitwise一致した。127は選択境界の対照であり、既定採用範囲を広げる証拠ではない。未実施の全model・全shape品質、MTP有効時の性能、R9700採用範囲はこの分類へ含めない。
- Phase83.5: 両targetの8192／128・MTP幅2・既存candidate併用探索を終え、上記exact geometry、query>=128、sliding／explicit score scaleなしのMXFP8 E4に限って未設定時の既定選択を実装中。Rust検証とnative dispatchを同時に変更する。新buildでの公開経路確認と正式性能測定は未完了であり、[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)で追跡する。
- rollback: Phase83.5の既定化後は`SLLM_CAUSAL_ATTENTION_GQA6_QTILE4=0`でgeneric MXFP8経路へ戻す。明示`=1`の既存opt-in範囲を維持し、その他の値は新しい既定選択を無効にする。未設定での既定化は上記scopeだけであり、非対応target／encodingの扱いとforce baselineを維持する。

### OUT-2026-09-09-P835-GQA6-QTILE8-W16: GQA6 QTILE8／16-wave（QTILE4比N0・検証中）

- scope: exact `gfx1030`、standard MXFP8 E4 KV、q24／kv4／d256、query count>=128、start position>=1024、sliding／explicit score scaleなし。QTILE4 flag未設定時だけQ4より優先する。gfx1201も同じscopeへ拡張する作業中であり、新buildの公開経路は未検証。
- 数値分類: QTILE4比**N0**。queryごとの8積のpair tree、wave shuffle、key順序、online softmax、weighted-V更新とBF16 RNEを維持する。query tileを4→8、wave数を8→16へ増やし、waveあたり3queryを維持してK/Vの共有loadを減らす。generic経路比の分類は上記Q4のN1を引き継ぐ。
- 検証: scratchのV620-Bでprefix1024／7168とquery127／128／129／1024の8ケースが独立oracle最大1 BF16 ULP、Q4比bitwise一致。127はscratch境界対照でありproduction採用外。両targetのproduction r2 release buildは成功したが、公開runtime経由の数値・モデル性能・正式反復は検証中。全入力のbitwise一致やBF16 full-model品質同等性の実測とは扱わない。
- gfx1201追加探索: M127／128／129×prefix1023／1024／1025とM1024／prefix7168の10ケースは独立oracle最大1 ULP、Q4比全出力bitwise一致、repeat一致。長prefixの単体時間は83.524→78.258 ms。境界の時間比は0.911〜1.008で、全shapeの改善を主張しない。証拠は`.local-artifacts/phase83-5/qtile8-r9700/`。
- rollback: `SLLM_CAUSAL_ATTENTION_GQA6_QTILE4=1`で従来Q4、`=0`でgenericへ戻す。その他の明示値とforce baselineも新しい既定選択を無効にする。

### OUT-2026-09-09-P83-NVFP4-PREFILL-COMPENSATED: ID87／ID89 NVFP4 prefill補償加算（N1候補・Phase83.5で既定化検証中）

- scope: Qwen3.8 NVFP4 W4A4の実測projection shape、`M>1`、`(K,N)=(5120,17408)`または`(17408,5120)`、有限で実用的な通常値のencoded入力。別の`K/N`、非有限payload、全model品質への一般化は行わない。
- baseline/candidate: Phase 82通常経路のID59 `sllm_matmul_nvfp4_w4a4_block16_prefill_row8_tiled256_v1`を正式baselineとする。ID59は同じE2M1 block16項を8個のK-strided partialと固定wave mergeへ渡す。ID87は同じblock16 integer dotとscale順を64x64/K32 tileで計算し、各FP32出力へKahan補償を加える。ID89はgfx1201のE2M1→E4M3FN exact ingressと固定FP32 WMMA fragmentを使い、fragment項をKahan補償付きで蓄積する。両candidateのglobal tensor scaleはID59と同じ`(accumulator * weight_tensor_scale) * input_tensor_scale`順で適用する。
- 分類: **N1候補（記載scope内）**。有限・実用的な通常値では、E2M1 block16 dotとE4M3 scaleのblock termを同じ集合として扱える。ID59の加算深さは概ね`ceil(K/256)-1+8`（`K=5120`で27、`K=17408`で75）であるのに対し、ID87のKahan誤差は標準的に`(2u+O((K/16)u^2)) * Σ|block_term|`へ局所化できる。ID89もexact ingress、固定FP32 fragment、Kahan outer accumulationへ差を局所化する。この記録はpointwiseなBF16改善、bitwise一致、全shapeの一律boundを保証しない。fragment内のboundを含むscope外の一般化は別途確認する。
- 数値検証: ID87のgfx1030 standalone probeは、非整列を含むsmall shapeの全点とlarge shapeのsampleを独立encoded long-double oracleへ照合し、`max_bf16_ulp=0`、repeat、finite、cleanupを確認した。ID89のR9700 gfx1201 probeは`M=63/64/65/1024`、wide/down、2 seedの10 caseで`max_candidate_oracle_ulp=0`、candidate repeat PASS、controlとの差は最大1 ULPだった。後者のcontrolはID64の診断比較であり、Phase 82正式baseline ID59との採否比較へ読み替えない。
- 性能・採否: ID89を明示選択したQwen3.8 8192/128、MXFP8、MTP無効、QTILE4併用のR9700探索行はprefill `364.0048`／decode `9.7603` tok/sだった。この値は当時のID89 includeでpragma scopeを修正する前の探索binaryであり、最終sourceの採用証拠へ再利用しない。単回探索であり、MTP目標、正式反復、ID59との差分帰属を示さない。Phase83完了時はID87／ID89ともproduction既定化を保留した。Phase83.5では両targetの同一fixture・MTP探索のHIP-only／非有限logitなし／cleanup zeroを確認し、下記の既定化を検証している。
- Phase83.5の既定化scope: exact gfx1030のID87／gfx1201のID89、`M>=64`、`(K,N)=(5120,17408)/(17408,5120)`。既存の明示opt-in範囲は変更しない。新buildでの通常CLI/API dispatchと正式反復は未完了。
- rollback: ID87は`SLLM_NVFP4_W4A4_PREFILL_FORCE_COMPENSATED=0`、ID89は`SLLM_NVFP4_W4A4_PREFILL_FORCE_WMMA_COMPENSATED=0`。NVFP4 prefill制御の明示値があれば新しい未設定時の既定化を抑制し、既存の明示provider選択順を維持する。
- 詳細: [Phase 83計画](../plans/archive/2026/09/1-10/phase83-mxfp8-fixed-sampling-mtp.md)、ID89 probe summary: `.local-artifacts/phase83/wmma-kahan-gfx1201-r1/summary.txt`。

### OUT-2026-09-09-P835-NVFP4-STAGE64: ID89 K64 staging（StageK32比N0・統合中）

- scope: exact gfx1201、NVFP4 W4A4、M>=256、K/N=5120/17408または17408/5120。logical ID89を維持し、別device symbol `sllm_nvfp4_w4a4_prefill_gfx1201_wmma128x64_kahan_stage64_v1`を選ぶ。
- 数値分類: StageK32比N0。K16ごとのWMMA項とKahan補償加算の順序、scale、BF16丸めを維持し、LDSへ先読みする範囲だけを増やす。
- 検証: R9700 scratchのM255／256／257・両tupleでsample oracle最大0 ULP、有限値、repeat、StageK32との全出力hash一致。M255は非選択側の対照。production compileでStageK64は127 VGPR、15,360 bytes LDS、spill0。r5公開runtimeではM255／256／257の新旧device symbol切替、5×5境界sample oracle最大0 ULP、全出力finite／repeat／cleanupをPASSした。r5全モデル単回はprefill383.547／decode13.794 tok/s、r2と128出力token一致。正式反復は未完了。
- 初回失敗: scale stagingの後半未初期化をstrided loadで修正した。K32／K64を同じwrapperへ入れた実装もLDS合算のため撤回し、別symbolへ分離した。証拠は`.local-artifacts/phase83-5/nvfp4-prefill-experiment/`。

### OUT-2026-09-09-P835-NVFP4-VGPR-REUSE: ID94 small-M weight再利用（ID88／ID90比N0・統合中）

- scope: exact gfx1030／gfx1201、NVFP4 W4A4、M2〜4、K/N=5120/17408または17408/5120。未設定時にgfx1030はID88、gfx1201はID90からID94へ切り替える。各targetの明示ROWGRID=1は従来ID88／90、=0等は新既定を無効にする。
- 数値分類: ID88／ID90比N0。同一laneの重みdecode／scaleを複数rowのregister accumulatorで再利用する。rowごとのdot4、FMA、wave reductionとBF16丸めを維持し、weight用LDS／Kごとのworkgroup barrierを増やさない。
- 検証: V620-B scratchの両tuple×M2〜4で全出力独立oracle／control／repeat最大0 ULP、有限値。200ms程度のwarmup後の20組AB／BAで1.104〜1.443倍。.local-artifacts/phase83-5/nvfp4-smallm-public-r5/のfresh公開APIテストでも6ケースの全出力独立oracle最大0 ULP、repeat／finite／ID94／symbol／grid／cleanupがPASSした。V620-A r5全モデル単回は197.570／17.806 tok/s、r4と128出力token一致。正式反復は未完了。scratch証拠は`.local-artifacts/phase83-5/nvfp4-smallm-vgpr-paired/`。

- gfx1201追加: 200ms連続warmup＋20組AB／BAの6ケースで1.671〜2.816倍、全出力独立oracle／ID90比較／repeat最大0 ULP。ID90と同じblock昇順、4回dot4、block単位FMA、wave32 shuffle、BF16 RNEを保つN0として対象2tuple・M2〜4へ統合した。両target compile-onlyとselector hostはPASS。fresh公開GPUと全モデルは次buildで確認する。

### OUT-2026-09-09-P83-NVFP4-SMALL-M-ROWGRID: ID88／ID90 small-M rowgrid（N1・ID84とのrowwise N0、Phase83.5で既定化検証中）

- scope: NVFP4 W4A4の実測projection shape、`M=2..4`、`(K,N)=(5120,17408)`または`(17408,5120)`。ID88はgfx1030、ID90はgfx1201だけを対象とし、他shape・targetへ一般化しない。
- baseline/candidate: ID88は各`blockIdx.y` rowをID84のM=1 body（gfx1030のID73 reduction）へ渡す既存rowgrid経路で、ID90は同じrow分割をgfx1201のID84 activation-shared M=1 body（ID67 reduction）へ渡す。各rowの入力・scale・weight・output offset、dot順、FP32 reduction、BF16 RNEはID84 M=1と同じである。
- 分類: **N1（記載scope内。ID84のM=1繰返しに対してはN0）**。Phase 82正式baseline ID59の各出力rowも同じblock16項の和であり、ID59は`tile_k=256`ごとに各partialへ`ceil(K/256)`項を逐次加算した後、wave shuffle 5段と固定merge 3段を通る。したがって加算依存深さは`K=5120`で`19-1+5+3=27`、`K=17408`で`68-1+5+3=75`である。ID88/ID90は`B=K/16`個のblockを各laneの`lane+32*j`順に処理し（lookaheadはこの順序を分割するだけ）、各laneの項数はそれぞれ10／34、wave shuffleは5段なので深さは両Kでそれぞれ`10-1+5=14`、`34-1+5=38`となる。各blockの4 DP4Aはint32で厳密に合計され、NVFP4の有限な実用E4M3 scaleと`0.25`を含むblock termはこのscopeではFP32に厳密に表現できるため、candidateの標準boundはID59の`gamma_27`／`gamma_75`以下の`gamma_14`／`gamma_38`（共通の`Σ|block_term|`に対するbound）となる。よってID59に対するworst-case誤差boundは増加しない。これはpointwiseなBF16改善、bitwise一致、全shape・非有限payloadへの一般化を保証しない。
- 数値検証: ID88のgfx1030 public/runtime probeは両NVFP4 tuple、`M=1..4`、repeat・finite・cleanup・独立sample oracleをPASSし、最大BF16 ULP差は0だった。ID90のR9700 public API probeも両tupleの`M=1..4`とFP8 controlを同一runtimeで実行し、ID90 dispatch、fallback未使用、`max_bf16_ulp=0`、cleanupを確認した。これはoperator-levelの証拠であり、full-model出力の証拠ではない。
- 性能・採否: ID88のV620 probeはwideのM2/3/4が`0.330887/0.464328/0.604291 ms`、downが`0.347286/0.489568/0.613569 ms`だった。ID90のR9700 probeはwideが`0.264843/0.272364/0.352605 ms`、downが`0.193083/0.275764/0.357604 ms`だった。測定は2 warmup＋5 measuredのbounded probeであり、新しい性能gateやfull-model採用条件を作らない。Phase83完了時は両IDをopt-inとして保持した。Phase83.5では上記exact target／M2..4／2 tupleの未設定時既定化を実装中で、新buildの公開経路確認は未完了。
- rollback: ID88は`SLLM_NVFP4_W4A4_SMALL_M_ROWGRID=0`、ID90は`SLLM_NVFP4_W4A4_SMALL_M_ROWGRID_GFX1201=0`。明示prefill制御は既定化より優先し、範囲外shapeは既存selectorへ戻す。
- 詳細: [Phase 83計画](../plans/archive/2026/09/1-10/phase83-mxfp8-fixed-sampling-mtp.md)、ID90 probe summary: `.local-artifacts/phase83/small-m-gfx1201-r1/summary.txt`。

### OUT-2026-09-09-P83-MXFP8-ATTENTION-STAGED32: ID93 staged32 attention（N1・Phase83.5で既定化検証中）

- scope: exact `gfx1030`／`gfx1201`、standard OCP MXFP8 E4 KV、`q_heads=24`、`kv_heads=4`、`head_dim=256`、
  query count `M=1..4`、committed KV length `>=1024`、sliding windowなし、explicit score scaleなし。
  それ以外のtarget、encoding、shape、長さは既存providerへ戻す。
- baseline/candidate: baselineのpacked causal attentionは1 request/queryのonline softmaxを維持する。
  ID93は同じMXFP8 codec、QK dot、`rsqrtf(256)`、key順のonline max／denominator／weighted-V更新を32の連続区間へ分割し、
  request-owned FP32 workspaceでstage1 partialを生成して区間順にstage2 mergeする。candidateのworkspaceは
  `24*32*(256+2)*sizeof(float)=792,576` byte/query（`M=4`で3,170,304 byte）であり、常駐FP16 mirrorは作らない。
- 分類: **N1（scope内）**。有限で実用的な通常入力ではmax比較を同じscore列へ局所化でき、区間内online recurrenceと
  ordered mergeの加算深さは、`K>=256`でstage8の概略`ceil(K/8)-1+7`からstage32の
  `ceil(K/32)-1+31`へ非増加となる。signed weighted-Vは`Σ|weight*V|`の絶対値boundで扱う。
  `expf`の有限通常値に対する有界誤差を仮定した分類であり、pointwise BF16改善、bitwise一致、非有限値への一般化は主張しない。
- 数値検証: V620 `gfx1030`のlength `8191/8192/8193`×`M=1/3/4`および`1023/1024/1025`×`M=1`、
  R9700 `gfx1201`の同形状fixtureで、独立scalar MXFP8/BF16 oracleは最大1 ULP、repeatは再現した。
  M=3/4ではbaselineとの差が最大1 ULPとなるためbitwise gateにはしない。V620測定のlength8192は
  stage32がM=1/3/4で約`0.949/1.548/2.388 ms`、既存controlが約`3.318/9.970/13.314 ms`だった。
  公開runtimeのV620-Bでもlength `1023/1024/1025`×`M=1/3/4`とlength `8192`×`M=3`を実行し、
  境界1023ではbaseline ID3、1024以上ではID93・2 dispatch・workspace・cleanupを確認した。
  これらはoperator-levelの探索値であり、full-model/API/MTP性能の完了証拠ではない。
- 採否: Phase83ではID93を両target共通の明示opt-inとして保持した。Phase83.5では上記scopeに限って
  `SLLM_CAUSAL_ATTENTION_DECODE_WAVE_STAGED32`未設定時の既定化を実装中。両targetのMTP full-model探索では
  HIP-only、非有限logitなし、cleanup zeroを確認したが、新buildでの既定dispatchと正式反復は未完了。
  明示`=0`（および`1`以外の値）、force baseline、範囲外shapeは従来経路を維持する。
  gfx1030のstage8明示選択はstage32未設定時に優先し、stage32明示`=1`の従来の優先順位を維持する。
- 詳細: [Phase 83計画](../plans/archive/2026/09/1-10/phase83-mxfp8-fixed-sampling-mtp.md)、
  V620 staged32 report: `.local-artifacts/phase83/attention-stage32-scratch-gfx1030-r1/report.md`。

### OUT-2026-09-08-P82-DEFAULT-SCOPE: Phase82の条件付き既定採用と保留

- scope: NVFP4 activation quantizer wave8（exact compile target `gfx1030`／`gfx1201`）、NVFP4 decode ID67（`M=1`、`1024<=K<=17408`、`K%16=0`、`N>=1024`、両target）、NVFP4 decode ID84（両targetの exact `(K,N)=(5120,17408)/(17408,5120)`）、FP8 outer decode ID82（exact `gfx1030`の4 tuple）、gfx1030 Qwen GDN row32（`M=1,qk/value=16/48,head_dim=128`）、Qwen3.8の既存projection/deferred/Graph/FP16 chain scope。各経路のtarget、shape、encoding、adapter、artifact gateは[Phase82 scope history](../history/2026/09/1-10/phase82-default-adoption-scope.md)に記録する。
- classification: wave8 quantizer、ID84、ID82、GDN row32、Qwen execution controlは**N0**。ID67は並列reduction treeへ変わるため**N1**。N0/N1はいずれも記載scopeだけに適用し、全model・全shapeへの一般化やtoken一致だけの品質主張はしない。
- precedence/rollback: unsetは上記の範囲だけ既定ON。明示force `=1` は対応shapeで優先し、`=0`／未知値はbaselineまたはfallbackへ戻す。`SLLM_*_FORCE_BASELINE=1`は最優先rollback。ID84を抑制したgfx1030 exact tupleは、他のwave4／activation-shared controlが未設定ならID73、gfx1201はID67へ戻る。別controlが設定された場合はsource上の明示優先順位に従い、ID82の`SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_LDS_LUT=0`／未知値はdirect controlが未設定なら採用済みID68へ戻り、scalar baselineは`SLLM_FP8_OUTER_DECODE_FORCE_BASELINE=1`で選ぶ。ID82の範囲外は既存経路を維持する。Qwen Graphはdeferredのstateless M1 decodeだけをcaptureし、terminal sampler・stateful KV/attentionはcapture外に残す。
- HOLD: GQA6 P64/P128、GQA6 rocBLAS F32、NVFP4 ID62/ID64/ID72は既定化しない。P64 partitionとP128はN2 scope、rocBLAS F32はprovider/reduction順変更、ID62はcorrected ID59より長いKの逐次block加算で誤差上限が増える既知のN2（K5120で319対約28段、K17408で1087対約76段。host stress sample最大絶対差0.75）、ID64は既定範囲不足、ID72はN2の丸め差と性能・ユーザー判断不足が理由である。ID62のGPU runnerやfull-model captureは人のN2判断を補助する任意証拠であり、追加の全model gateではない。P64のFP16 LDS表現だけはN0だが、partition採用を意味しない。
- verification: NVFP4 quantizerは両targetの26 fixture／5 mode NumPy oracle、ID84 projection packは両targetのbitwise／repeat／`max_bf16_ulp=0`／cleanup、ID82は既存のexact tuple oracleと境界検査、GDN row32はstate/output oracleとcleanup、Qwen coreはhost focused testsと既存target-scoped GPU evidenceを確認した。R9700 Gemmaの新NV67既定はbaselineと3 sampling modeのtoken hashが異なる一方、HIP実行／finite／fallbackなし／cleanup 0はPASSしたため、N1 target拡張を全model出力同値とは扱わない。最終r7 source identityがcurrent source anchorであることは[Phase82 evidence ledger](../history/2026/09/1-10/phase82-optimization-evidence.json)の`final_build_reuse_mapping`に記録する。未実施の全model・全shape性能は採用根拠に含めない。
- rollback identity: `SLLM_NVFP4_ACTIVATION_QUANTIZE_WAVE8`、NVFP4 decode controls、`SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_LDS_LUT`／`SLLM_FP8_OUTER_DECODE_FORCE_BASELINE`、`SLLM_LINEAR_ATTENTION_GFX1030_ROW32_LDS`、Qwen target-scoped `0`／malformed controls、および既存attention baseline flagsを維持する。候補固有削除は[Phase82 cleanup history](../history/2026/09/1-10/phase82-optimization-cleanup-default-adoption.md)へ分離する。

### OUT-2026-09-07-P79-DECODE-DEFAULTS: NVFP4/FP8共通decodeの条件付き既定化（N1）

- scope: exact gfx1030、M1。NVFP4 ID67はK1024..17408・K%16=0・N>=1024、
  FP8 E4M3FN outer ID68はK128..17408・K%64=0・N>=64。layout等の既存public契約を維持する。
- 分類: **N1**。実数式、量子化recipe、scale、BF16 RNEを維持。有限encoded入力で、
  FP8 ingressはFP16へexactに展開され、加算依存深さはscalarのKから最大概ね`8*ceil(K/256)+5`へ減る。
  NVFP4はblock16内をexact integer dot4で合計し、scale適用後のFP32加算深さを
  `ceil(K/256)+8`から`ceil(K/512)+5`へ減らす。追加の再量子化・非決定atomicはない。
- correctness: production ABIの実Gemma形状、中間形状、採用境界で独立encoded oracle、
  repeat、全BF16比較を確認。実モデルはfinite、HIP-only、cleanup0。
- 出力影響: 候補別の探索で、最初の生成差はNV wave4が0始まり10、FP8 dword8が1。
  固定入力logitsも異なる（最大KLDは各0.347253/5.834114）。生成列・品質の同等性は主張せず、
  演算の誤差非増加に基づいて数値gateを自動承認する。
- 性能/採否: 採用。最終Gemma比較のdecodeは17入力で1.5516→15.5622、65入力で1.5445→15.3531 tok/s。
  これは両selectorとprojection共有の合成効果で、個別候補の寄与率ではない。個別効果は探索・operator表へ分離した。
- rollback: `SLLM_NVFP4_W4A4_DECODE_FORCE_DP4A_WAVE4=0` と
  `SLLM_FP8_OUTER_DECODE_FORCE_GFX1030_DWORD8=0`。範囲外/他targetの既定は維持。
- 詳細: [採用範囲](../history/2026/09/1-10/phase79-adoption-scope.json)、[Phase79履歴](../history/2026/09/1-10/phase79-common-optimization.md)。

### OUT-2026-09-07-P79-SHARED-EXECUTION: projection共有と実行制御の共通化（N0）

- scope: 共通prepared completion/Graph制御とモデルadapter、HIP gfx1030/gfx1201の適合NVFP4 gate/up。
  同一activation、encoding/layout、verified input scale bitsの一致を要求し、非対応targetは通常matmulへ戻す。
- 分類: **N0**。同一quantizerの出力を2 projectionで共有し、member別weight/scaleと既存consumerを維持。
  completion/Graphはownerと同期方式の変更で、演算・丸め順は変えない。request専用queueとupload完了順序を維持する。
- correctness: Gemmaの両targetで共有ON/OFF・deferred ON/OFFが全12位置の全logits一致。
  各caseのdecodeはpack144 submission、prefill0、無効時0。native Gemma形状の3-node Graph replayも独立oracle一致。
  Qwen3.8 Graph ON/OFFとQwen4B FP16/MXFP8 KVのprofiled/deferredは生成列一致、HIP-only、cleanup0。
- 採否: 適合Gemma packは既定採用。Qwenの既存個別opt-in、共通deferredのopt-inは維持。
  Gemmaの追加deferred効果は測定ばらつき程度であり、全モデルの新しい既定へは拡張しない。
- rollback: `SLLM_PREPARED_PROJECTION_SHARING=0`、`SLLM_PREPARED_DEFERRED_COMPLETION=0`。
  Qwenの既存deferred既定を止める場合は優先する`SLLM_QWEN_DEFERRED_COMPLETION=0`を使う。
- 詳細: [runtime契約](../architecture/runtime.md)、[Phase79履歴](../history/2026/09/1-10/phase79-common-optimization.md)。

### OUT-2026-09-07-P79-NVFP4-PREFILL-REDUCTION: 基準加算順の復元（N1／基準互換N0）

- scope: NVFP4 W4A4 ID59、M>1/K>0/K%16=0/N>0、有限encoded値。
- change: 各row waveの長い逐次和を8個のK-strided partialへ分け、ID11と同じwave treeと8-wave merge順へ戻す。
  重みのshared predecode、activation量子化recipe、tensor scale、BF16 RNEは維持する。
- 分類: **旧ID59→修正ID59はN1**。FP32加算深さを `ceil(K/32)+5` から
  `ceil(K/256)+8` 以下へ減らす。**ID11→修正ID59は有限encoded入力でN0**。
  E2M1×E4M3の積とterm積はFP32で厳密であり、scale因数分解は追加丸めを生じない。
  NaN payload互換は主張しない。
- 観測: 旧ID59はGemma M65でID11比最大KLD 2.144631/maxabs 8.718750だった。
  修正版gfx1030 draft2はM63/64/65のtiny/実shape全BF16出力、独立oracle、repeatがPASS。
  Gemma M3/17/65と各3 teacher-forced decodeの12位置でID11と全logits一致（KLD/maxabs 0）。
  finite、実HIP dispatch、fallbackなし、cleanup0を確認した。品質評価一般のPASSには読み替えない。
- 状態: 採用。修正後の両targetで固定入力12位置がID11と一致し、共通化後の最終buildでも確認した。source/build/raw identityは下記履歴で追跡する。
- rollback: `SLLM_NVFP4_W4A4_FORCE_BASELINE=1` はID11を選ぶ。ただし既存ID11は
  M4096/N4096でlaunch status260を観測しており、大規模prefillの万能切戻しとは扱わない。
  ID59修正版はrow共有gridを維持し、この制約の回避を維持する。
- 詳細: [Phase79履歴](../history/2026/09/1-10/phase79-common-optimization.md)。

### OUT-2026-09-05-P78-FP8-PREFILL-LOAD64: V620 FP8 prefill協調load（N0）

- scope: exact gfx1030 ID71、M1024/K6144/N5120およびM1024/K5120/N10240。
- change: 各threadへ隣接8-byteのFP8値を割り当て、2本の32-bit loadを1本の64-bit loadへまとめる。
  low/high wordを同じ変換helperへ渡し、LDSの最終配置・half2 dot順・FP32加算・scale・BF16 RNEを維持する**N0**。
  A/Wが8-byte非整列の場合は既存full-tile body、それ以外の形状も既存経路を使う。
- correctness: 両形状4-copy全BF16出力・repeat・FP64 tile境界期待値と、A/W offset1/2/3/4/7 fallback、
  guard／finite／checked cleanupをPASS。本番matmul TUの直接compile G1でも同じ比較をPASSし、実行前後SHAは不変。
  両target release buildを確認した。gfx1201に新経路を選択する変更はない。
- performance: private copy別中央値平均は`3826.618→3569.996 us`と`6134.939→5683.695 us`。
  r25 V620長文1 warm＋3 measuredはr24と全token／文章／停止理由／audit一致、HIP-only／cleanup0。
  prefill `308.308→312.115 tok/s`、TTFT約373 ms短縮、E2E約401 ms短縮を確認したが、Phase78目標は未達。
- identity: source manifest `12d409672170b2d3bbfa5db90a8afaf4d879a7b62463972796f668c3553514c3`、
  gfx1030 archive `f30b4921c755f2e3297ab069cff1dc696c355b0b933a0f8afba8c35a26c26aa7`。
- rollback: r24のfull-tile body。既存ID71 opt-in内に限定する。
- 詳細: [Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-FP8-GDN-Z-TUPLE: V620 GDN z投影の形状定数化（N0）

- scope: exact gfx1030、Qwen3.8 GDN zのM1/K5120/N6144、既存ID82 opt-in内だけ。
- change: 既存rolled P2 tuple bodyへ形状を定数として渡し、generic bodyの境界判定・address計算を削減する。
  LUT ingress、各laneのK順、4つのdot2、FP32加算・wave reduction、scaleとBF16 RNEを維持するため**N0**。
- correctness: private 4-copy比較で全6144出力一致、各copy64境界点の独立FP64期待値、両provider repeat、
  guard／finite／checked cleanupをPASS。本番r24 archiveのGDN共有・個別matmul・Graph replay数値比較もPASS。
  source／binaryの実行前後SHA不変、native host testと両target release buildをPASS。
- performance: privateのcopy別中央値平均は`0.090251→0.081091 ms`。VGPRは57→170、active blocksは8→2だが、
  実測時間は約10.1%短縮した。実モデルの効果は別途記録し、private速度をwhole-modelへ読み替えない。
  r24短文／長文は各1 warm＋3 measuredでr23と全token／文章／停止理由／audit一致、HIP-only／cleanup 0。
  長文decode `15.402→15.412 tok/s`はMAD範囲内であり、whole-model改善を確定せずopt-in候補に留める。
- identity: source manifest `65a600a23cb5be3bc52f1ddf490f6234364ef5b7744b85a9dc7137ce843ce4c7`、
  gfx1030 archive `0bb737e39826cd17a358fff757f56c7c60ecea0eb6b46e36ce9b20bf07400a1b`。
- rollback: r23のgeneric ID82 body。既存ID82 opt-in自体の省略も従来どおり利用できる。
- 詳細: [Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-FP8-GDN-SHARED: GDN qkv/zの量子化共有（N0・opt-in）

- scope: exact Qwen3.8のGDN 48 pair、gfx1030／gfx1201、M1 K5120、N10240／6144。
- change: 同じBF16 input viewを既存FP8 quantizerで一度だけ量子化し、bytesとrow scaleを2つの既存matmulで共有する。
  member別weight scale、provider／hipBLASLt algorithm、演算順とBF16出力丸めを維持する。
- 分類: **N0**。quantizerはweight非依存であり、全Kのmax／448とE4M3FN丸めを変更しない。
  request-owned workspaceは5124 bytes、M>1は従来どおり2 matmulへ分解する。
- correctness: r23の両target G1でmember別weight・scaleの独立期待値、全BF16出力一致、入力変更後比較、
  実repeat、3-node HIP Graph capture／replay、cleanup 0をPASS。共有3 dispatch、個別合計4 dispatch。
  source／archive／binaryの実行前後SHA一致を確認した。短文17/17と長文9435/128は両target各1 warm＋3 measuredで
  baselineと全token／文章／停止理由一致、HIP-only／cleanup 0。長文decodeはV620 `15.266→15.402 tok/s`、
  R9700 `18.719→18.828 tok/s`。最終4行3 warm＋10 measuredの性能比較は未完了。
- identity: source manifest `75c0177bb01da93e1efaaa4a6944eebccd7d89840c0f5f4881573c6aa518fe53`。
  archiveはgfx1030 `402a213f60731a59b453af6985b1f4551f9103db6fd974c94cc1950ac5551d35`、
  gfx1201 `50da39102fbdd5093c8a173fa4d65d37e7e91d7a12bd148cfa2a67a743d99c71`。
- 決定: `SLLM_QWEN38_FP8_GDN_PROJECTION_PACK2=1`のopt-in候補。省略がrollbackで、既存NVFP4 opt-inとは独立。
- 詳細: [Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-V620-PIPELINE: NVFP4次stage先読み（N0）

- scope: exact gfx1030、ID62、M1024/K5120/N17408だけ。single LDSの64x64/K32 tileを維持し、
  次stageのpacked activation／weightとscaleを現在stageのdot4演算中に読む。
  rawからLDSへの変換、block16演算順、scale乗算、FP32加算、BF16 RNEを維持するため**N0**。
- private correctness/resource: tiny M32/33/65 K48/N131全点FP64、M1024 wide/down全BF16一致＋
  64点FP64、sentinel付きrepeat、finite／cleanupをPASS。VGPR86／LDS6144／scratch 0／active 5。
- performance: 4-copy／128共通prewarm／3 warm＋10交互measuredでwide
  `8229.996→7391.363 us`（MAD `8.170/44.446 us`）。downは約8.2%遅いため適用しない。
  独立global symbolとexact shape判定を接続済み。r21 production native launcher G1で
  tiny M33/65全点FP64、実wide全17,825,792出力一致＋64点FP64、sentinel付きrepeat、finite／cleanupをPASS。
  exact／neighbor／transposed形状のmetadata判定も確認した。r21長文9435/128は旧r17と全token・文章・audit一致、
  HIP-only／nonfinite 0／cleanup 0。prefill `300.074→307.808 tok/s`、TTFT `30671.365 ms`。
  decodeは`15.266 tok/s`、TPOT `65.504 ms`。1 warm＋3 measuredの探索結果で、最終目標は未達。
  旧Index32 bodyとそれ以外のrouteを維持し、rollbackはr19のgfx1030 provider。

### OUT-2026-09-05-P78-R9700-LOAD-SHARING: 短文load指定とdecode activation共有（N0・短文実モデル確認済み）

- scope: exact gfx1201。ID64はM17/K5120/N17408のpacked A/Wを通常loadへ変更し、
  128x64/K32 tile、WMMA contribution、scale乗算、FP32加算、BF16 RNEを維持する。
  M17/K17408/N5120の既存split4 partialも同じload指定変更だけを行い、分割・reduction順は維持する。
  ID84はM1の(K,N)=(5120,17408)/(17408,5120)だけactivationをLDSへ共有し、
  P4 weight先読み、dot4順、FMA、wave reduction、tensor scale、BF16 RNEを維持する。双方**N0**。
- private correctness: ID64はtiny M17K48N37と実shapeの全BF16一致、64点FP64 oracle、
  repeat／finite／cleanup 56/56をPASS。ID84修正版r4はtiny全点FP64、実2shape全BF16一致、
  control/candidate双方のrepeat／finite／cleanup 168/168をPASSした。初回r3は測定実装不備で採用証拠から除外。
- private performance: 4-copy／128共通prewarm／3 warm＋10交互measuredで、ID64実shapeは
  `928.828→659.646 us`（MAD `3.020/9.301 us`）。ID84は通常中央値で
  wide `105.401→103.781 us`、down `107.166→105.786 us`。tinyは遅いため適用しない。
- integration: 独立device symbolとexact shape分岐を接続し、r20 gfx1201 release buildをPASS。
  r20 binary SHA-256
  `af92677c86c5c6ddefb1ab9c14f2d45d32b517e26bc504cfca368f76530bbdd1`。
  ID64 wideのproduction launcher G1はtiny／実shape全BF16一致、FP64／repeat／cleanup 56/56をPASS。
  ID84のr20 archive単独linkによるnative launcher G1はtiny／neighborの旧symbolと実2shapeの新symbolを確認し、
  全BF16一致、tiny全点／実shape64点FP64、repeat／finite／cleanup 224/224をPASSした。
  split4 ordinaryの私有比較も全BF16一致、64点FP64、双方repeat／cleanup 40/40をPASSし、約33%短縮した。
  r21 split4 native launcher G1も同じcorrectnessをPASS。r21実モデル17/17は旧r19と全token・文章・audit一致、
  HIP-only／nonfinite 0／cleanup 0。TTFT `236.314→179.119 ms`、prefill `79.325→107.655 tok/s`。
  長文9435/128も旧r15と全token・文章・audit一致、HIP-only／nonfinite 0／cleanup 0。
  prefill `1150.989 tok/s`、decode `18.719 tok/s`、TPOT `53.422 ms`。
  1 warm＋3 measuredの探索結果であり、decode目標と正式反復の確認は残る。
  rollbackはr19。ID72 N2の採用保留とPhase 78の既存完了条件を維持する。

### OUT-2026-09-05-P78-REQUEST-GRAPH: 要求開始時のhostコピー削減（N0）

- scope: residentからの要求生成。既存の3種類のrewrite gateがfalseなら所有済みgraphをmoveし、
  無効な変換による深いcloneを省く。WeightLoadPlanはresident/core間で不変のArc共有とする。
  graph/layoutの検証、adapter・環境条件の評価、要求ごとのstate／queue生成、GPU演算は維持するため**N0**。
- correctness: resident関連12件とresidual fusion 1件のhost test、両target release buildをPASS。
  r18/r19の両target短文17/17は旧版と全token一致、HIP-only／nonfinite 0／cleanup 0。
- performance: r18 gfx1201はsetup `26.255→21.664 ms`、TTFT `241.619→236.740 ms`。
  r19 gfx1030はsetup `23.431 ms`、TTFT `234.249 ms`、decode `16.186 tok/s`。
  r19 gfx1201はsetup `22.027 ms`、TTFT `236.314 ms`、decode `19.560 tok/s`。
  計画共有だけの追加速度差はばらつき内である。各1 warm＋3 measuredの探索値で、
  V620のhost時間差はばらつきを含み、確定改善率は主張しない。
  r18 gfx1030途中測定は別buildと重なったためhost性能判断に使わない。
- rollback: r17の要求開始経路。r19 binary SHA-256はgfx1030
  `90cf96f3cd5fe3abaa297496e1a73f7a4143651fbbde3143964c51331d76d162`、gfx1201
  `672a0581c4d389497b30b513a9082c232b45454de052e7fa2fc37b7c134b2dcb`。

### OUT-2026-09-05-P78-FP8-DECODE-TUPLE: gfx1030 ID82の3形状専用経路（N0）

- scope: M1、(K,N)=(5120,17408)/(6144,5120)/(5120,10240)。定数K/Nとrolled loopで
  境界判定を除き、LUT、load、dot、reduction、FP32 scale、BF16 RNEの順を維持するため**N0**。
  3つの独立global symbolから共通template bodyを使い、他shapeの旧ID82 bodyは維持する。
- correctness/resource: r17 production launcherの3形状と非一致K64/N32で、全BF16 bit一致、
  独立host prefix oracle、全256 E4M3 ingress、repeat、finite、cleanupをPASS。
  公開dispatch metadataも形状に対応したsymbolを確認。専用kernelはVGPR170／LDS544 bytes／
  scratch 0／active blocks 2、旧kernelはVGPR57／active blocks 8で、資源増加を専用symbol内に限定する。
  archive SHA-256は`1d286790bbbd0149de8de3b0623a4c9598556b086689b3287b8d97705b98fee1`。
- performance: private固定候補のK6144/N5120とK5120/N10240は、各launch前に計測外で128 MiBを
  読み同期する共通条件で、中央値`169.201→133.881 us`／`229.642→197.641 us`。
  r17 G1のK5120両形状はMADが差より大きく、その速度比だけでは改善を確定しない。
  r17実モデル9435/128（1 warm＋3 measured）はr15と全生成token一致、HIP-only／nonfinite 0／cleanup 0。
  decode `14.793→15.224 tok/s`（約2.9%改善）、TPOT `67.602→65.686 ms`、
  prefill `299.134→300.074 tok/s`。Phase 78の最終性能条件は未達である。
  candidate binary SHA-256は`15b3b3ef3ed9bec265a84efd916aadb6b92e01b8002769488b2e7f3079ad52d3`。
- rollback: r15の旧ID82 body。binary SHA-256
  `9bd0f0de7293cecb9aecdadd09a2555371ec49ff82093766875a30c872b280aa`。

### OUT-2026-09-05-P78-FP8-FULL-TILE: gfx1030 ID71の境界判定削減（N0）

- scope: M1024、(K,N)=(6144,5120)/(5120,10240)だけ、既存64x64/K32 tileの境界判定を省く。
  FP8→FP16 ingress、dot2の順、FP32 scale、BF16 RNEを維持するため**N0**。
  それ以外は旧bodyを使用し、既存ID71 symbolと同じ共有メモリを使う。
- correctness/resource: r15 production launcher対private旧bodyで、実2shapeとM219の全BF16 bit一致、
  FP64 oracle、全256 E4M3 ingress、repeat、cleanupをPASS。LDS8704 bytes、VGPR52、spill 0、
  active blocks 7を確認した。最初の二重LDS案は接続前に修正している。
  archive SHA-256は`aac223c6c805113ff0011c34aa2178bc1261d279b951ed52608215b18a9c306c`。
- output/performance: G1は`3581.838→3426.796 us`／`5681.006→5398.762 us`、M219はほぼ同じ。
  r15長文9435/128はr14 Graph-onと全生成tokenが一致し、HIP-only、nonfinite 0、cleanup 0をPASS。
  prefill `296.763→299.134 tok/s`、TTFT `31820.643→31565.963 ms`。
  decodeの観測値は`14.884→14.793 tok/s`であり、decodeの改善は主張しない。
  各1 warm＋3 measuredの探索値で、Phase 78の最終条件は未達。
  rollbackはr14 binary `6df7be6c22eb327173c9f0147a86807fd153113e47b3f7ff8e0e9a7841ca065c`。

### OUT-2026-09-05-P78-GRAPH-SPAN: gfx1030 stateless decode Graph（N0・opt-in）

- scope: exact Qwen3.8 M1 decodeの同一layer内stateless区間を、requestが保持する同じprepared plan／bufferで
  captureする。初回はeager、以後は同じkernel列をreplayし、stateful演算とterminal処理は区間外へ残す。
  式、dtype、演算順、丸めstageを変えないため**N0**。capture自体は出力を実行・更新しない。
- correctness: gfx1030 M1K35N37のRMSNorm＋BF16 matmul G1で5入力のeager／graph／数値oracle bit一致、
  capture時出力不変、1,000 replay、早期解放BUSY、cleanup 0をPASSした。
  r14 archive SHA-256は`454d1bba2796fdd589c4f8476b06ffc5726c69596fd1cc2058cc7c8c04166aad`。
- output: 同一r14 binaryの17/17と9435/128 off/onで全生成tokenが一致し、最初の分岐はない。
  logical submission、kernel identity/countも一致、HIP-only、nonfinite 0、cleanup 0、各build内再現性をPASS。
  各requestは128 span／1,176 kernel nodeを保持し、replayは短文1,920回、長文16,128回だった。
  binary SHA-256は`6df7be6c22eb327173c9f0147a86807fd153113e47b3f7ff8e0e9a7841ca065c`。
- performance: 作成費用込みの短文TPOTは`62.693→63.487 ms`で約1.3%退行、長文は
  `68.094→67.186 ms`、decode `14.686→14.884 tok/s`で約1.35%改善。各1 warm＋3 measuredの探索値であり、
  Phase 78の最終性能条件は未達。長文向けopt-in候補とし、短文での速度改善は主張しない。
  rollbackは`SLLM_QWEN38_GFX1030_GRAPH_SPANS`の省略または`0`。r14時点ではR9700未検証。
- r15でgfx1201の既存prepared FP8 Lt planへ同じcapture機構を接続した。rank／algorithmは変更しない。
  M1K6144N5120でcapture時出力不変、7入力のeager bit一致、1,000 replay、cleanup 0をPASS。
  R9700の17/17 off/onも全token・logical kernel監査が一致し、HIP-only／cleanup 0を確認した。
  TPOT `51.273→51.260 ms`、decode `19.503→19.508 tok/s`で、短文の速度改善は確認できない。
  長文9435/128も全token／logical kernel監査が一致し、TPOT `54.993→53.856 ms`、decode
  `18.184→18.568 tok/s`（約2.1%改善）、HIP-only／cleanup 0を確認した。各1 warm＋3 measuredの探索値。
  長文は既存ID72条件を保ったGraphのみの比較で、ID72のN2採用判断待ちは維持する。
  opt-in／rollbackは`SLLM_QWEN38_GFX1201_GRAPH_SPANS`。
  archive SHA-256は`cb9d40ac30287911f5aceaa35b7e1f5b54b03b3123c01c3b6139f074d1942dab`。

### OUT-2026-09-05-P78-FP8-SCHEDULE: gfx1030短文tileとID82 P2先読み（N0）

- scope: ID71 FP8 outer prefillの実測済みK/N・M=2..32を32x32／32x64 tileへ変更し、
  ID82 decodeではraw activation/weightを2反復分先読みする。FP16へのexact ingress、
  各出力のdot2順、wave reduction、FP32 outer scale、BF16 RNEは維持するため**N0**。
  それ以外のprefill shapeは既存64x64を維持し、公開grid metadataも同じshape判定へ揃えた。
- correctness: 最新production launch wrapperへ再リンクし、短文M17/31/32/33とtinyの
  oracle・全BF16一致を確認。ID82 P2は全256 E4M3 code、N37/N67の境界fixture、
  全8実shape×4 weight copiesで全BF16一致、finite、反復決定性、cleanupをPASSした。
  r7 archive SHA-256は`ae5e3c362806d9ff320f79a3169c4063d9cbd5fba96f027a0c912d1132e2cf01`。
  r9では未適用だった6実shapeへ短文tileを拡張し、production wrapper対private referenceの
  11 shape（M17/31/32/33と非整列tiny）で全BF16一致・oracle・cleanupをPASSした。
  r9 archive SHA-256は`cab4f8512c40b7563e212d1c1995ddddcea5fe688c1db31881e72cc0a1b39352`。
- output/performance: v3-r7の全4行でv3-r6と生成tokenが一致し、HIP-only、terminal nonfinite 0、cleanup 0。
  長文decodeは`13.647→14.658 tok/s`、TPOTは`73.279→68.220 ms`へ改善した。
  短文tileの効果はv3-r6に含む。いずれも最終3 warm＋10 measuredではなく各1 warm＋3 measuredの探索値。
  r9の17/17はr8と生成tokenが一致し、prefill `61.497 tok/s`、TTFT `328.600→304.616 ms`。
  M33以上とwhitelist外のshapeは既存64x64へ戻す。
- source/rollback: `native/hip/src/fp8_prefill_short_m32.inc`と`matmul_kernel.hip.cpp`。
  ID82の変更前archiveは`3201f34562edbc334cb37ab4aba1ae53674ec0c848fe0a8c2e83dc168a79dce0`、
  短文tileの変更前archiveは`a1d7a769ea7268e604f8212e61b1e4ddc96f2f6fee8f91e30d6c79b250c27d9d`。
  詳細は[Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-NVFP4-SCHEDULE: gfx1030短文tileとdecode P2先読み（N0）

- scope: gfx1030 NVFP4 W4A4。ID62のM=2..32では64行tileの無効な後半32行を除き、
  ID84 decodeではweight/scaleのraw bitsを2反復分先読みする。
  block16内の整数dot、FP32 scale適用、block間加算、wave reduction、BF16丸めの順序は維持するため**N0**。
  行同士の加算は存在せず、raw bits先読みも数値変換を追加しない。
- correctness: 変更前archive `a1d7a769ea7268e604f8212e61b1e4ddc96f2f6fee8f91e30d6c79b250c27d9d`
  と短文prototypeのM17/31/32/33、wide/down全BF16一致、tiny oracleをPASSした。
  統合後archive `3201f34562edbc334cb37ab4aba1ae53674ec0c848fe0a8c2e83dc168a79dce0`
  のproduction launch wrapperへ再リンクし、M2/16/17/31/32/33・K48/N37の全点oracle、
  実shape全BF16一致、cleanup failure 0を確認。decode P2もscale/tiny oracleとwide/down全BF16一致を確認した。
  targetはV620 PCI03、ROCR index 1。実モデルtokenと性能はv3-r6で確認中。
- performance/decision: 短文prototypeのM17 wide/down時間比は`0.746／0.854`。
  M33は退行するため短文tileを選ばない。P2の既存単体加重改善は約`1.095x`であり、
  実モデル全体の改善とは扱わない。rollbackは既存64行tileと同include内のbaseline ID73 body。
- source: `native/hip/src/matmul_kernel.hip.cpp`、`native/hip/src/nvfp4_decode_scale_lut.inc`。
  詳細は[Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-FP8-SHORT-RANK: M17のhipBLASLt選択（N1・combined model checkpoint）

- 対象: gfx1201 E4M3FN、M17/K6144/N5120だけ。zero-workspace heuristicを4件要求しrank3を選ぶ。
  M1 decodeの32件要求・環境overrideは維持する。
- 解析: control 123381／candidate 123380は同じCOMPUTE32F、FP32 WMMA、GSU1、LocalSplitU1。
  ISAでFP16 accumulatorとatomicがないことを確認した。同じFP8積のFP32丸め和として、
  標準の共通上界`gamma_6143 * sum(abs(products))`を共有するN1とする。
  tight boundの改善、pointwise誤差非増加、全入力bit一致は主張しない。
- 根拠: [TensileのGSU定義](https://github.com/ROCm/rocm-libraries/blob/develop/shared/tensile/Tensile/Common.py)、
  [AMDのTensile解説](https://rocm.blogs.amd.com/artificial-intelligence/reverse-hipblaslt-tensilelite/README.html)。
  両symbolを確認したROCm 7.14 code object SHA-256は
  `dfb7d5c735a92e23e3e9fb7ee0c6dc1f350844db22a3c2e92ef8053821ec4526`。
- 検証: private probeの全BF16 outputはcontrol／CPU oracleと一致、repeat／cleanup成功。
  単体中央値`0.115561→0.091641 ms`。production rank境界host検証はPASS。
  R9700 r13の17/17実モデルは1 warm＋3 measuredでPASSし、TTFT `241.01478 ms`、
  HIP-only、nonfinite 0、cleanup成功、反復生成一致を確認した。BF16 thin／NVFP4 split4も同時に含むため、
  速度・出力への単独寄与は未分離である。r11との最初の差は15番目の出力token（20→15）。
  証拠は`.local-artifacts/phase78-resume/fp8-gfx1201-m17-k6144-n5120-run.log`。
- rollback: 当該M17 policyを従来single-result/rank0へ戻す。r13 archive SHA-256は
  `9ec38a7c76602d3c162b69d72d29e7a8efd3f36b82d25e413e8ad14e7ec51120`。

### OUT-2026-09-05-P78-LOAD-HINT: prefillの通常load（N0）

- 対象: gfx1030 NVFP4 ID62 Index32 longと、FP8 ID71 long 64x64。
- 変更: activation／weightのnontemporal load hintを通常loadへ変更する。
  FP8はpacked/scalar計4箇所、NVFP4はpacked 2箇所。算術・codec・scale・tile・index幅は維持する。
- 検証: NVFP4は固定r10/r11の別binaryによる7 shape全BF16 dumpが一致し、tiny oracleをPASS。
  FP8は旧production対private候補、新production対同じprivate候補で7 shapeの全BF16一致とoracleをPASS。
  r11のV620全4行はr10と生成token一致、HIP-only、nonfinite 0、cleanup成功。
- identity: r11 gfx1030 archive
  `7e3e2720f7f238a2df90161fdeb7df572251d24254a7617fc26a77227df2913a`。
  証拠は`.local-artifacts/phase78-resume/nvfp4-index32-r10-r11-dump-summary.txt`と
  `fp8-gfx1030-half2-64x64-no-nt-r11-summary.txt`。
- 性能: r11探索測定の長文prefillは`283.231→296.387 tok/s`。最終性能条件は未達。
  短文NVFP4と特定shapeのFP8 decodeへ広げたr12はG1・生成token一致をPASSしたが、
  実モデル改善を確認できず採用を取り消した。このr11の採用範囲には含めない。

### OUT-2026-09-05-P78-NVFP4-INDEX32: 安全な32-bit offset（N0）

- 対象: gfx1030 ID62、M33以上かつ全tensor extentが32-bit内に収まるshape。
  端数tileのoffset余裕もguardし、対象外は従来64-bit kernelを使用する。
- 変更: offset/index幅だけを縮小し、tile、codec、scale、積和順、BF16変換を維持する。
- 検証: guard境界host検証、production GPUのM32/33/65・K48/N131 oracleと
  4実shapeの全output bit一致をPASS。r10全4行の生成tokenはr8と一致。
- identity: gfx1030 r10 archive
  `e7e6ee6e62651d81933e7fe494849ef0b68d7229958d0a2c083d8a2e714d527d`。
  証拠は`.local-artifacts/phase78-resume/nvfp4-index32-r10-production-g1*`。

### OUT-2026-09-05-P78-GDN-REGISTER: 短文recurrent stateの保持（N0）

- 対象: gfx1030/gfx1201、Qwen qk16/value48/head128、M2..32。
- 変更: 各threadのFP32 stateをtoken間でregister arrayへ保持する。
  scalar演算、reduction、物理state index、BF16出力変換は変更しない。
- 検証: 両GPUのproduction launchでM2/17/32/33 × zero/nonzero、全786,432 FP32 stateと
  BF16 outputがgeneric referenceとbit一致、cleanup成功。M33のgeneric分岐も確認した。
  r10のV620全4行とR9700短文で生成token一致、HIP-only、nonfinite 0、cleanup成功。
- identity: gfx1030 archiveは上記r10、gfx1201は
  `70f6c56ce2b0b8dd35cda5678bb6a0133abeebf441bb0672f9318b3ccf4acb84`。
  証拠は`.local-artifacts/phase78-resume/linear-gdn-register-state-r10-production-g1-summary.md`。
- 性能: 単体M17はV620約4.15倍、R9700約2.9倍。実モデル短文TTFTは
  V620 `285.236 ms`、R9700 `247.610 ms`。探索測定であり最終速度ゲートは未達。

### OUT-2026-09-05-P78-NVFP4-QUARTER: ID62のscale乗算移動（N0）

- scope: NVFP4 DP4A prefillのactivation scaleをLDSへ置く際に`0.25F`を掛け、
  各outputのblock sum側にあった同じ係数を除く。整数dot、weight scale、block間加算、BF16丸めは維持する。
  block16の整数dot絶対値は2304以下で、有限E4M3 scaleとの最初の積はFP32で厳密表現できる。
  二進の係数移動で後段のoperandは変わらないため**N0**。
- correctness: dot範囲と全256 E4M3符号の1,179,904組をhostで確認し、非NaN値のbit不一致0。
  両側NaNは同値扱いで、NaN payloadの一致はこの証明の対象外とする。
  V620 PCI43ではproduction archiveと旧演算のprivate referenceを比較し、M65/K48/N131の全点oracle、
  M128/1024のwide/down全BF16一致をPASSした。旧archiveとprivate referenceの比較もPASSし、
  scale loadのcache修飾とE4M3 decodeを揃えている。
  gfx1201はstrict compile-onlyをPASSした。これはgfx1201 GPU実行の証拠とは扱わない。
- identity: r8 gfx1030 archive SHA-256は
  `063d1001642b67882f15340e8b632b8615aec51a932054fc8b9b7d0e97974778`。
  ローカル証拠は`.local-artifacts/phase78-resume/nvfp4-quarter-g1-{old,r8}.log`。
- performance: 実shapeの単体時間は同時比較の旧演算から約3〜4%短縮した。
  v3-r8の全4行・各1 warm＋3 measuredでr7の生成tokenと一致、HIP-only、nonfinite 0、cleanup 0。
  長文prefillは`275.307→279.797 tok/s`へ約1.6%改善したが目標未達。decodeは`14.684 tok/s`でほぼ横ばい。
  rollbackはactivation scale stagingの係数を外し、block sum側へ戻す。

### OUT-2026-09-05-P78-P64-LDS: P64 attentionのFP16 LDSとnative変換（N0）

- scope: gfx1030/gfx1201のQwen3.8 GQA6 FP16 KV decode、partition 64、tile16。
  K/VをFP32へ変換してからLDSへ置く処理を、元のFP16 bitsをLDSへ置き利用時に同じFP32値へ変換する処理へ変更。
  FP16の有限値・subnormal・signed zeroはFP32で厳密表現できる。Inf/NaNは既存の符号・payload変換を明示保持する。
  partition、QK加算、softmax、V加算、merge、BF16丸めの順序は不変なので**N0**とする。
- correctness: native変換は両targetの全65,536 FP16符号で独立host oracleとbitwise一致。
  P64 prototypeはL=8191/8192/8193/9435×2 seedでproduction controlとの全BF16一致、
  FP64 stable-softmax oracle、repeat、nonfinite確認をPASSした。
- performance/resource: LDSは32 KiBから16 KiB、workspaceは不変。
  gfx1201 prototypeの8条件合計時間比は`0.820`。gfx1030のproduction実モデルv3-r5は
  9435/128の生成tokenがcontrolと一致し、decode `13.700 tok/s`、runtime cleanup成功。
  gfx1201もproduction再リンクG1をPASSし、v3-r6長文decodeは`17.607→18.092 tok/s`へ改善、
  同一行の生成token一致とruntime cleanup成功を確認した。単体の比率を全体改善とは扱わない。
- identity/rollback: sourceは`native/hip/src/causal_attention_kernel.hip.cpp`。
  gfx1201の変更前archive SHA-256は`9fa32e05bdfd764c96ee7085e023771068a60d39871adce0a1b5a290e0f55612`。
  rollbackは同関数のFP32 LDS branchであり、P128へ切り替えない。
  詳細は[Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-P128: gfx1030 GQA6 decode partition変更（N2・採用保留）

- scope: exact `gfx1030`、Qwen3.8 GQA6（query 24／KV 4 head、head dimension 256）、FP16 KV decode。
  P64からP128へpartitionを増やし、同じsoftmax／weighted V式のFP32部分和とmerge順序が変わる。
- classification: **N2**。現行production archiveへリンクした独立FP64 stable-softmax oracleで、
  L=8191/8192/8193/9435とseed 0/7919の8条件を検証した。P64はFP64 oracleのBF16丸めと全点一致、
  P128は合計4点で最大1 BF16 ULP差があった。例えばL8193／seed0のL2誤差は
  `9.752086430e-3→9.752088890e-3`、最大絶対誤差は`2.441160609e-4→2.441651891e-4`となった。
  この有限な観測誤差をworst-case上限の証明や実文章品質の判定へ一般化しない。
- numerical/resource: 両providerとも8条件の全反復がbitwise決定的、oracle／actual nonfinite 0、probe exit 0。
  targetはV620 PCI `0000:03:00.0`。archive SHA-256
  `af30b49d4dd7f813f3a91f303eab3a4e92a04499a57598cd1208e531ff0d3707`、probe SHA-256
  `29e305843234c944ddd7ea352f343f834932b1b1825454eb8f64d2367dfa7c1b`。
- output/decision: 既存長文探索ではP128だけを外すと5個目からの生成token分岐が解消した。
  P128は採用保留を維持し、Phase 78の継続測定にはP64を使う。P128採用をPhase完了の前提にしない。
  rollbackは`SLLM_CAUSAL_ATTENTION_GQA6_DECODE_SPLIT_P128`を外し、P64 opt-inを維持する。
- evidence: `.local-artifacts/phase78-resume/p64-p128-current-pci03-*`。
  詳細は[Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-ID72: gfx1201 NVFP4のFP16 staging（N2・判断待ち）

- scope: exact `gfx1201`、固定Qwen3.8-27B mixed-NVFP4のW4A4 prefill。
  ID64のblock16 WMMA＋FP32 scale適用から、ID72のexact FP16展開＋full-K FP32 GEMMへ切り替える。
  E2M1とE4M3 scaleの積はFP16で正確に表現できるが、積和の順序は変わる。
- classification: **N2**。生成tokenの変化に加え、独立FP64 scalar oracleで誤差増加を確認したためN1とはしない。
  M=17/127/128/129/219、K/N=5120/17408と17408/5120、rocBLAS solution 0/136321を調べた。
  positive finite E4M3の全127 codeと符号付きE2M1から合成入力を作り、各caseのBF16不一致点の先頭256点をFP64と比較した。
  例えばsolution 136321のwide形状では、確認点の最大`abs(error)/sum(abs(terms))`が
  ID64の`0.000748140701`からID72の`0.000748232564`へ増加した。既存oracleの許容範囲内だが、
  この差分点を選んだ標本は全出力の平均誤差や実文章品質の評価ではなく、worst-case上限の証明でもない。
- performance/output: 端数219-token chunkをID64で補完した1 warmup＋3 measuredの長文prefillは
  `522.150→1134.906 tok/s`、HIP-only／fallback 0／cleanup 0。benchmark v2の長文生成hashは
  `03a85b481c6a52b8bc029f3959a9b42987a46697a1b048e292ced4944984adf3`から
  `628ad99500e71de824a1e06fb918ec03a400587f7f9337ecc3225c07cbb591fb`へ変化した。
- decision: 2026-09-05に、この具体的な速度・丸め差を示してユーザー判断を依頼した。回答待ちで既定採用しない。
  rollbackは`SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_F16_STAGING`を外し、既存ID64を選ぶ。
- evidence: Git管理外`.local-artifacts/phase78-resume/id72-short-shapes.cpp`／同`.log`、
  `r9700-id72-tail.json`。詳細と最終採否は[Phase 76〜81計画](../plans/active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)。

### OUT-2026-09-05-P78-BF16-GDN-THIN: M17 GDN薄型投影のrow-wise reduction（N1・production G1 PASS、combined model checkpoint）

- scope: exact `gfx1030`／`gfx1201`、Qwen GDN thin projectionの`M=17,K=5120,N=48`だけを対象とする。
  `K=70/71,N=47`のtiny確認はguard外の既存tiled16経路であり、新providerのproduction scopeには含めない。
- baseline/candidate: baselineは既存`matmul.bf16_fp32.tiled16.v2`の16x16 tile内K逐次FP32 accumulation。
  candidateは既存`matmul_bf16_decode_body<32U,8U>`を各`(row,column)` blockへ呼ぶ`grid=(N,M)` providerである。
  BF16 input、paired load、各FP32 product、wave32 tree、8-wave fixed tree、BF16 RNE outputを維持し、runtime ID／public ABIは追加しない。
- classification: **N1**。`K=5120`のcandidateは各threadが10 paired loads（20 product terms）を処理し、局所加算深さ19、wave32 tree深さ5、8 partialの有効な最終tree深さ3となる。
  構造上の深さは`27`、zero laneを保守的に含めても`<=29`で、baselineのK逐次深さ`5119`より小さい。
  同じ入力項、BF16→FP32 product、FP32 accumulator、BF16 RNE stageを維持し、race、atomic、未初期化値、fallbackはない。
  このboundはworst-case誤差上界の非増加を示すもので、pointwise非増加や全入力bit一致は主張しない。
- correctness/output: r13 archiveのpublic `PrefillTiled16` launcherを使ったproduction G1をV620 PCI `0000:03:00.0`とR9700 PCI `0000:07:00.0`で実行した。
  tiny K70/71を含む両target全shapeで独立FP64 oracle mismatch 0、production/reference BF16 diff 0、repeat mismatch 0、finite、cleanupをPASSした。
  実modelの個別kernel帰属は未分離であり、combined checkpointを下記へ記録する。
- performance/resource: common prewarm 128、warmup 3、measured 10のmean event時間は、V620が
  `0.190954→0.016484 ms`、R9700が`0.178870→0.017640 ms`（private tiled16 reference→production row-wise）だった。
  これは単体meanであり、実model速度の単独帰属ではない。
- combined model checkpoint: V620 r13（BF16 thinとNVFP4 split4を同時に含む）17/17は1 warm＋3 measuredで
  prefill `79.749667 tok/s`、`213.167033 ms`、TTFT `240.039708 ms`（r11 `281.070 ms`）、decode `15.826946 tok/s`だった。
  HIP fallback 0、nonfinite 0、cleanup 0、deterministic true。r11との差はzero-based 14（15番目、`15→20`）から分岐した。
  R9700 r13も17/17 PASS、prefill `79.014116 tok/s`、`215.151429 ms`、TTFT `241.01478 ms`（r11 `244.851 ms`）、decode `19.538382 tok/s`だった。
  RもHIP fallback 0、nonfinite 0、cleanup 0、deterministic trueで、r11との差はzero-based 14（15番目、`20→15`）から分岐した。
  V/Rは同じ位置で逆向きのtoken差になった。この分岐をBF16 thin単体またはsplit4単体へ帰属させず、両targetのcombined N1 output effectとして記録する。
- source/rollback/identity: sourceは`native/hip/src/matmul_kernel.hip.cpp`。rollbackはexact shape guardを外して既存tiled16へ戻す。
  r13 gfx1030 archive SHA-256は`9bbec6f8cd40c13934768037c90adb6320f25c4b3a7dba8da0eed9d75e1d0dd7`、
  gfx1201は`9ec38a7c76602d3c162b69d72d29e7a8efd3f36b82d25e413e8ad14e7ec51120`。
  evidenceは`.local-artifacts/phase78-resume/bf16-gdn-thin-production-g1-summary.md`と同`*-run-gfx1030.log`／`*-run-gfx1201.log`。

### OUT-2026-09-05-P78-NVFP4-SPLIT4: M17 stage split-K=4 reduction（N1・production G1 PASS、combined model checkpoint）

- scope: exact `gfx1030`ではID62の`M=17,K=5120,N=17408` wideと`M=17,K=17408,N=5120` down、
  exact `gfx1201`ではID64の`M=17,K=17408,N=5120` downだけを対象とする。tiny `M=17,K=48,N=37`とR9700 wideはguardで拒否する。
- baseline/candidate: scaled block16 terms、E2M1→E4M3 ingress、FP32 partial、tensor scale、BF16 RNEを維持する。
  candidateはK stageを4 contiguous partitionsへ分け、`p0,p1,p2,p3`を固定順でpartial-reduceし、最後にtensor scaleを一度だけ適用してBF16 RNEする。
- classification: **N1**。`B=K/16`、`S=ceil(B/2)`とすると、K5120は各partition 40 stage／80 termsで
  `79+3=82`、K17408は136 stage／272 termsで`271+3=274`のcandidate加算深さとなる。
  unsplit baselineはそれぞれ`B-1=319`／`1088-1=1087`、tiny K48はcandidate／baselineとも`2`である。
  標準boundは`gamma_82`対`gamma_319`、`gamma_274`対`gamma_1087`で非増加だが、pointwise非増加やbitwise equalityは主張しない。
- correctness/output: production G1はV620 PCI `0000:03:00.0`とR9700 PCI `0000:07:00.0`でPASSした。
  V620 wide/downはBF16 diff 0、FP64 oracle、repeat、nonfinite、cleanupをPASSした。R9700 downはBF16 diff 30をN1 report-onlyとし、
  control/candidateともFP64 oracle mismatch 0（max normalized error `0.0001631567`）、repeat、finite、NaN fixture（両方5120 nonfinite）、
  cleanup 16 alloc/freeをPASSした。bit diff 30は誤差boundを無効にするpointwise主張へ昇格しない。
- performance/resource: V620 wideは`401.064→392.204 us`、downは`758.727→499.205 us`、R9700 downは
  `1024.648→905.488 us`（各control→split4 production G1）。V620 split4はVGPR54／LDS4608／scratch0／active blocks 8、
  R9700 split4はVGPR97／LDS7680／scratch0／active blocks/CU 7、active waves/CU 56、occupancy 0.875だった。
  V620/R9700のcombined r13 model checkpointは上記BF16 thin entryに記録した。BF16 thin／split4のどちらがtoken分岐へ寄与したかは未分離であり、
  combined N1 output effectとして扱う。この単体結果だけで個別candidateのmodel帰属や性能採用完了とはしない。
- source/rollback/evidence: split4は既存ID62/ID64のexact shape guard内に限定し、scope外は従来providerへ戻す。
  production G1 evidenceは`.local-artifacts/phase78-resume/nvfp4-gfx1030-split4-production-g1-pci03.log`と
  `nvfp4-gfx1201-split4-production-g1-pci07.log`。tiny拒否、target identity、固定partial reduction、scale一回適用を同ログで確認した。

### OUT-2026-09-02-P73-GFX1201-MXFP8-WIDE-N: production selectorのN<=32768拡張（N1）

- scope: exact `gfx1201`、OCP MXFP8 E4M3 W8A8 block32/E8M0 prefill。ID31／34／36／37のN上限だけを
  16,384から32,768へ広げ、既存M/K、64／128列alignment、他target／format／decodeを維持する。
- arithmetic/classification: kernel、量子化recipe、E4M3 value、E8M0 scale、FP32 accumulation、BF16 RNEは変更しない。
  row8から既存WMMA treeへ選択が変わり得るため、Phase 63と同じ**N1**として扱う。
- verification/decision: N=17,408／32,000／32,768のhost selection、32,769／32,832のfallback、prepared providerと
  既存gfx1201 codec/provider testをPASSした。ユーザー指示により新規範囲のGPU数値oracle、生成token、性能測定は省略し、
  未実施を明記したうえでselector scopeを採用する。
- rollback: 3 predicateの上限を16,384へ戻す。
- details: [Phase 73履歴](../history/2026/09/1-10/phase73-gfx1201-mxfp8-wide-n-selector.md)と
  [追跡要約](../../ci/matrix/phase73-gfx1201-mxfp8-wide-n-selector-v1.json)。

### OUT-2026-09-02-P72-GFX1201-MXFP6-WIDE-N: ID45のN<=32768拡張（N1）

- scope: exact `gfx1201`、OCP MXFP6 E3M2 W6A6 block32/E8M0 prefill matmul。Phase 70 ID45のN上限だけを
  16,384から32,768へ広げる。M>=17、K>=2048、K%32=0、N>=1024、他target／format／decodeのcomplementは維持する。
- baseline/candidate: baselineはN>16,384をID25 tiled16へ戻す。candidateは既存ID45のE3M2→E4M3 exact ingress、
  K16 FP8 WMMA×2、E8M0 scale、block間FP32 accumulation、BF16 RNEをそのまま広幅Nへ選ぶ。量子化recipeやkernel算術は変更しない。
- classification: **N1**。Phase 70と同じ固定WMMA treeへのprovider切替であり、real-number equation、入力集合、dtype、
  scale、accumulator、丸めstageを維持する。測定8 shapeではID25／ID29とBF16 digestまで一致したが、全入力bit一致とは主張しない。
- correctness/output: N=16,384/16,385/17,408/17,409/24,576/32,000/32,767/32,768の各45 sampled pointを
  独立FP32 oracleでPASSした。最大相対誤差`0.0036457598`、非有限不一致0、各5 row top-1とoutput digestは両controlに一致、
  repeat不一致0。Qwen3.5-27Bの4 sampleも生成`[23066,23066,23066,23066]`で一致した。
- performance/resource/decision: ID45はID25比`3.0731〜10.6190x`。強制指定なしのQwen3.5-27B 512-token prefillは
  旧既定`81.746517`から`383.170165 tok/s`へ4.6873倍となり、resident／peakは`24,115,002,880 / 24,777,018,880` byte、
  HIP-only、fallback／cleanup 0だった。検証結果に基づきN<=32,768を採用した。
- rollback: selector上限を16,384へ戻す。運用上のcontrolは`SLLM_MXFP6_PREFILL_FORCE_TILED16=1`。N=32,769以上は
  現状もID25へfail closedに戻る。
- details: [Phase 72履歴](../history/2026/09/1-10/phase72-gfx1201-mxfp6-wide-n-selector.md)と
  [追跡要約](../../ci/matrix/phase72-gfx1201-mxfp6-wide-n-selector-v1.json)。

### OUT-2026-09-02-P70-RDNA-MXFP6-VIA-E4M3: packed E3M2 ingressとgfx1201 WMMA（N0/N1）

- scope: OCP MXFP6 E3M2 W6A6 block32/E8M0のprefill matmul。exact `gfx1030`のID43は明示benchmark専用、
  exact `gfx1201`のID45は`M>=17`、`K>=2048`、`K%32=0`、`1024<=N<=16384`へ限定採用する。decode M=1、
  scope外shape、別target、KV default、量子化recipe、GGUF encoding、sampling、stop/usageは変更しない。
- baseline/candidate: ID43はID29 col8のrow/column/K分解、E8M0 scale、FP32 accumulation、wave reduction、BF16 RNEを維持し、
  packed E3M2を実数値exactなE4M3FN bitへ変換して既存E4 decodeへ渡す。gfx1201 ID44/45は同じvalue/scale byteから
  K32 tileだけをE4M3へmaterializeし、K16 FP8xFP8-to-FP32 WMMAを2回、scale pair、block間FP32 accumulation、BF16 RNEの
  固定treeで処理する。ID45はID44と同じ算術treeのまま、同じ3-byte groupの4値を一括変換・32-bit LDS storeする。
- classification: ID43とID44→ID45は**N0**。E3M2→E4M3FNは全64 codeで実数値exactで、ID43はID29とのBF16 digest、
  ID44/45は相互の演算順と丸めstageを維持する。従来ID29→gfx1201 WMMA familyは**N1**。実数式、入力項、scale、FP32
  accumulator、BF16 RNEを維持し、差を固定K16 WMMA treeへ局所化できる。逐次K32 dotより加算依存深さを増やさず、race、
  atomic、未初期化値、silent fallbackを使用しない。
- oracle/state: 全64 E3M2 code×4 packed laneをexact `gfx1030`／`gfx1201` device oracleでbit exactに確認した。ID43は
  production 5 shapeでID29 digest一致。ID44/45/46は独立FP32 oracle、非有限位置一致、repeat determinismをPASSし、
  P70-Fの最大相対誤差は`0.003875792259350419`だった。selector境界、prepare freeze、別target非選択、HIP-only、
  fallback false、cleanup 0も確認した。
- output/quality: 固定Qwen3.5-4B MXFP6、FP16 KV、512／2,048 inputのID44／45全sampleで生成tokenは
  `[23066,23066,23066,23066]`だった。旧providerとWMMA familyのlarge-M operator BF16 digestは異なる。full-model logitの
  最初の差、top-1、KLD、perplexityは未収集であるが、recipe不変のN1 arithmetic変更なので旧KV default用`0.99` gateは適用しない。
- performance/resource/decision: exact gfx1201のID44→ID45は3 warmup＋10 measuredで512 input
  `1276.494→2157.868 tok/s`（1.690倍）、2,048 input`1506.933→2423.308 tok/s`（1.608倍）。ID45はLDS 6,912 byte、
  SGPR/VGPR 38/115、spill/private 0でshape限定採用した。N128 ID46はVGPR 167かつ両full-model行でID45より遅く
  benchmark-only。gfx1030 ID43も512／2,048で約22.7%／21.6%遅くbenchmark-onlyとした。
- rollback: ID44は`SLLM_MXFP6_PREFILL_FORCE_PHASE70=gfx1201-n64`、従来tiled16は
  `SLLM_MXFP6_PREFILL_FORCE_TILED16=1`。scope外は従来providerを維持する。
- details: [Phase 70履歴](../history/2026/09/1-10/phase70-rdna-mxfp6-mxfp8-path-reuse.md)と
  [追跡要約](../../ci/matrix/phase70-rdna-mxfp6-mxfp8-path-reuse-v1.json)。

### OUT-2026-09-01-P67-GFX1030-MXFP8-MMQ: staged col8 scoped default（N0）

- scope: exact `gfx1030`、OCP MXFP8 E4M3 W8A8 block32/E8M0のprefill matmul。`M>=128, K>=2048, K%32=0`かつ
  `2560<=N<=16384`または`M>=512 && N==1024`だけ既存ID27 col8を選ぶ。短M、M<512のN=1024、未計測N、語彙head、別target、decode M=1は
  既存providerを維持する。量子化recipe、KV default、sampling、stop/usageは変更しない。
- change: gfx1201 ID37のN方向再利用をsoftware-decode gfx1030へ転用し、ID38 col16／ID39 col32を評価した。
  両候補は既存ID27をfull-modelで上回らず明示benchmark-onlyとし、追加shape sweepで境界を確定したID27だけを限定採用した。
- classification: **N0**。ID22/27/38/39は各outputのMXFP8 value／E8M0 scale、FP32 accumulator、加算順、wave reduction、
  BF16 RNE stageを維持する。18 case×10回と追加M=`512/2048` caseのBF16 output digestはprovider間で一致した。
- oracle/state: K=`31/32/33` admission、M=1、M=`17/127/128/129/512/2048`、N tail、N=`32/1024/2560/4096/8192/9216`、
  target非選択、override優先順位、prepare-time freezeをhost／exact gfx1030 GPUで確認した。HIP-only、fallback false、cleanup 0である。
- output/quality: 固定Qwen3.5-4B、FP16 KV、512／2,048 inputの全sampleで生成tokenは
  `[23066,23066,23066,23066]`だった。operatorでbit一致しておりrecipe不変のN0なので、KV形式変更用top-1 `0.99` gateは起動しない。
- performance/resource/decision: 同一最終binaryでrow8→scoped defaultは512 inputが
  `72.1830 -> 207.6111 tok/s`（2.8762x）、2,048 inputが`71.2428 -> 208.2710 tok/s`（2.9234x）。
  resident／peakは不変。ID27/38/39はLDS `8,704/17,152/34,048` byte、VGPR `46/42/83`、spill 0である。
- rollback: `SLLM_MXFP8_PREFILL_FORCE_ROW8=1`。scope外は既存row8、ID38/39は明示overrideだけである。
- details: [Phase 67履歴](../history/2026/09/1-10/phase67-gfx1030-mxfp8-tile-transfer.md)と
  [追跡要約](../../ci/matrix/phase67-gfx1030-mxfp8-tile-transfer-v1.json)。

### OUT-2026-09-01-P66-GFX1201-LOWP-PROVIDER: N128 matrixとtyped provider移植（N0）

- scope: exact `gfx1201`のMXFP8 E4M3 W8A8 ID37、MXFP6 E3M2 W6A6、NVFP4 W4A16／W4A4、
  MXFP4 W4A4 prepared routing、およびFP16／MXFP8 E4 KVのtyped causal-attention候補。量子化recipe、GGUF encoding、
  weight/KV default、sampling、stop/usageは変更しない。
- change: ID36の各output列と同じFP32 arithmetic treeをN128 tileへ広げるID37を追加し、format/block/layout／activation pack／
  tile／inner productをprepare時にfreezeする共通providerへ各形式を接続した。attentionはq4k4／q4k8／q8k8のtyped loadを追加した。
  NVFP4／MXFP4 W4A4は既存device kernelへのprovider routing移植であり、数値式の異なる別candidate kernelではない。
- classification: **N0**。ID37は独立output列の同時処理数だけを変え、各outputの項、scale、FP32 accumulator、加算tree、
  BF16 RNE stageを維持する。attention control/candidateの全output digestと最大absolute error 0、matrix ID36/37のBF16 digest、
  NVFP4／MXFP4 routingのoracleが一致した。provider freeze自体はdevice arithmeticを変更しない。
- oracle/state: MXFP8はM=`127/128/129`、N=`64/127/128/129/256/512/1024`とproduction shapeを実行し、
  K31／33は期待どおりhost rejection、K32はGPU受理を確認した。attentionはFP16／MXFP8 KVのM=`128/512/2048`、
  MXFP6／NVFP4／MXFP4は形式ごとの非整列blockとM>1をexact gfx1201で実行した。
  全採用evidenceはHIP-only、fallback false、cleanup 0である。
- output/quality: 同一入力の数値差、token/logit分岐は観測していない。N0かつquantization recipe不変なので旧KV default用
  top-1 `0.99` gateを起動しない。MXFP8 full-modelの4 outputは全run `[23066,23066,23066,23066]`だった。
- performance/resource/decision: ID37はwide/down operatorをID36比14.45%／6.27%短縮し、exact gfx1201、Phase 65 family、
  N%128=0へ限定採用した。LDS 1,024 byte、SGPR/VGPR 40/164、spill 0、wave32、WMMA 16命令である。
  attention候補は全primary rowで4.3〜27.3%遅くproduction不採用。MXFP6、NVFP4、MXFP4は形式別の既存kernel／fallbackを維持し、
  MXFP4 full MoE productionはscope外とした。
- rollback: ID37 scope外はID36／既存provider、attentionは既存q4k1等、各低精度形式は従来device kernelである。
  persistent BF16/FP32 weight展開、FP32 attention/KV planeは追加しない。
- details: [Phase 66履歴](../history/2026/09/1-10/phase66-gfx1201-reusable-low-precision-attention-transfer.md)と
  [追跡要約](../../ci/matrix/phase66-gfx1201-low-precision-provider-summary-v1.json)。

### OUT-2026-09-01-P63-GFX1201-MXFP8-WMMA: 大規模prefill WMMA provider（N1）

- scope: exact `gfx1201`、OCP MXFP8 E4M3 W8A8 block32/E8M0、M>=128、K>=2,048、1,024<=N<=16,384、
  K%32=0、N%64=0のprefill matmul。M=1、N=32／LM head、`gfx1030`、`gfx942`、未知targetは既存providerを維持する。
- change: resident value/scaleを直接読み、従来row8のK32逐次FP32 dotを、N64 tileあたり8個のK16
  FP8xFP8-to-FP32 WMMA、block scale適用、block間FP32 accumulationへ変更する。入力E4M3/E8M0 byte、実数式、項、scale、
  FP32 accumulator、BF16 RNE outputは同一である。K32ごとのFP32 contribution LDS store/readは行わない。
- classification: **N1**。固定K16 treeを2個使う加算依存深さは従来のK32逐次FP32和より増えず、term/scale/dtype/rounding stageを
  欠落させない。差はmatrix instruction内を含む固定FP32 treeへ局所化でき、race、atomic、未初期化、fallbackではない。
- oracle: exact gfx1201でcandidate 7 case／21 submissionを3 repeatし、M=`127/128/129`、wide/down/output、N=1,024、
  N=32／M=1非選択を確認した。最大production相対誤差は`0.0036960265`。special E4M3/E8M0 byteと13 oracle点は
  非有限4/4一致、mismatch 0、relative `0.0004885198`、repeat digest一致、HIP-only、fallback false、cleanup 0だった。
- output: 同一MXFP8 artifactのrow8/candidate 10 case／20 rowはtop-1 `19/20=0.95`、KLD mean `0.0029974001`、
  p99/max `0.0153089212`、perplexity相対差`-0.516618%`。最初のlogit差は`b255` prefill position 254 index 0
  `6.25→6.21875`、最初のtoken差は`b511` decode position 511 `13→220`だった。旧KV default判定のtop-1 `0.99` gateは適用しない。
- performance/resource/decision: 3+10の512/1,024/2,048/4,096 prefill中央値は
  `1,727.595/1,814.619/1,722.844/1,588.366 tok/s`。model residentは従来と同一で、persistent workspaceは追加しない。
  exact gfx1201の上記shapeだけへscoped default採用し、明示row8およびscope外row8をrollbackとする。
- details: [Phase 63履歴](../history/2026/09/1-10/phase63-gfx1201-mxfp-matrix-prefill.md)と
  [追跡要約](../../ci/matrix/phase63-gfx1201-mxfp8-wmma-prefill-v2.json)。

### OUT-2026-08-31-P62-LOWP-CODEC: 共通low-precision codecと起動境界specialization（N0）

- scope: MXFP8 E4M3／MXFP6 E3M2 W/A matmul、MXFP8 E4/E5 KV append/attention、NVFP4の共通scalar/block read/write、
  exact `gfx1030`／`gfx1201`。数値recipe、accumulator、丸めstage、public encoding、defaultは変更しない。
- change: E4M3FN/FNUZ、E5M2、E3M2、E2M1、E8M0とMX block 32／NV block 16を共通device-inline codecへ抽出し、
  attentionのruntime encoding switchをgeneric、decode wave、GQA shared/qtile、scaled long-prefillのkernel起動境界へ移した。
- classification: **N0**。W/A value/scale byteとM=`1/3/17`の6 BF16 output hash、KV append byte、29-case attention output hashを
  beforeとbit exactに維持した。実数式、FP32 accumulation、BF16 RNE、OCP scale/packing、NVFP4 outer scaleは同一である。
- oracle: 両GPUでdecode 1,104 code、encodeのzero/subnormal/tie/max/Inf/NaN境界、MX `31/32/33/256`、
  NV `15/16/17/256`を独立host oracleへ照合した。W/A/KV/full-attentionはHIP-only、fallback false、cleanup 0だった。
- output: 固定Qwen3.5-4Bの3／4および17／4 full-model試行はbefore/afterで生成token列が一致した。
  数値差、最初の分岐、品質recipe変更はないため追加quality gateを起動しない。
- performance/decision: 共通codecと起動境界specializationを両targetへshared adoptionした。代表17-token FP16-KV prefillは
  gfx1030 MXFP8/MXFP6 `47.31/98.23→48.48/99.23 tok/s`、gfx1201 `36.67/32.72→72.87/115.30 tok/s`。
  MXFP8 KV=8,193 attentionはgfx1030 `5.248→2.462 ms`、gfx1201 `3.515→1.569 ms`だった。
- rejected: cross-plan activation cacheはbuffer generation/liveness identityがなくstale readをfail-closeできず、単純fusionはN tileごとに
  activation量子化を重複するため不採用。rollbackはPhase 61までのconsumer-local codecであり、public rollback optionは追加しない。
- details: [Phase 62履歴](../history/2026/08/21-31/phase62-reusable-low-precision-block-optimization.md)。

### OUT-2026-08-30-MXFP8-E4-DEFAULT: block16廃止とstandard OCP MXFP8 E4既定化（N2）

- scope: reviewed Qwen3.5-4B BF16 dense text／full attention／single GPU／head dim 256、exact `gfx1030`、`gfx1201`、
  `gfx942:sramecc+:xnack-`。対象外model/laneのfixed recipeは変更しない。
- change: `kv-fp8-e4-block16`／`kv-fp8-e5-block16`を全production admission境界で拒否し、省略時を
  `kv-mxfp8-e4-v1`（OCP E4M3FN、block 32、E8M0）へ変更する。gfx942でもFNUZへ再解釈しない。
- classification: KV resident量子化とattention decode値がFP16から変わるためN2。明示`fp16`をrollbackとして残す。
- direct GPU: ROCm 7.14.0のV620 exact `gfx1030`とR9700 exact `gfx1201`で、head dim
  `31/32/33/255/256/257`のvalue／scale byte oracle、append 6、head dim 256のpacked direct attention 1をPASSした。
  いずれもHIP-only、fallback 0、cleanup 0である。gfx942実機は未実施であり、他tupleへ一般化しない。
- full-model draft quality: 同じQwen3.5-4B lockと10 case／20 logit rowの一回測定で、KLD p99は両targetとも
  `0.004945428206833837`、KV request-state peakは`68,354,048`から`60,195,840` byte（11.935%減）だった。
  top-1一致はgfx1030 `1.0`、gfx1201 `0.85`で、gfx1201はfreeze済み`>=0.99`を満たさない。これは実行correctnessと
  memory効果のdraft evidenceであり、品質gate PASSへ読み替えない。report SHA-256はgfx1030
  `d342b7755857848741d67d3ae37a580dcc6bb6442fabfbe49321fa03f013b9c2`、gfx1201
  `1280e6e0172bc25be1c69a370d6f449e8455c3d4d6108aaaa91f96d8b6e471c0`である。candidateは明示形式ではなく
  省略時resolverからtarget-aware graphへlowerした。
- decision: gfx1201品質未達を隠さずN2として保持した上で、2026-08-30のユーザー明示決定を採用根拠としてdefault変更を維持する。
  release品質昇格は主張せず、明示`fp16`をrollbackとする。
- history: Phase 53/54のblock16 correctness／quality／early-stop evidenceは削除せず、今回の採用根拠には使わない。

### OUT-2026-08-27-P53: KV FP8 block16 target別default判定（N2・旧recipe非採用、follow-up active）

- scope: reviewed Qwen3.5-4B BF16 dense text/full attention/single GPU、exact `gfx1201`の
  `kv-fp8-e4-block16`とexact `gfx1030`の`kv-fp8-e5-block16`。標準OCP `kv-mxfp8-e4`／
  `kv-mxfp8-e5`はreference-onlyのexplicit比較である。
- baseline/candidate: baselineはFP16 KV。block16はtoken内head-dimension方向16値ごと、標準MXFP8は32値ごとに
  独立E8M0 scaleを持つ。append/outputはBF16、attention accumulatorはFP32を維持するが、KV量子化recipeを
  変更するため**N2**とする。
- superseded recipe: 以下のcorrectness、quality、metrics、summary、digestは、有限値を飽和させない最小scaleを使った
  `kv-fp8-e4-block16-v1`／`kv-fp8-e5-block16-v1`へ結合する。このevidenceは監査履歴として保持するが、
  descriptor v2のcorrectness、品質、default採否を決めない。
- correctness: gfx1201／gfx1030でblock16とMXFP8のpadded value／scale byte oracle、append 6、direct attention 1、
  HIP-only、fallback 0、cleanup 0をPASSした。gfx942はfresh Phase 53 reportがなく、standard OCP MXFP8はFNUZ非互換のためunsupportedである。
- quality: FP16→block16→MXFP8をresident完全解放付きで3 repeatした。gfx1201 block16はKLD p99
  `0.0038687249522990803`、top-1 `0.85`、long-context loss `0.08333333333333337`、gfx1030 block16はKLD p99
  `0.04331390780013198`、top-1 `0.8`、long-context loss `0.16666666666666663`だった。全repeatは同値で、
  reference-only MXFP8 KLD p99はgfx1201 `0.004945428206833837`、gfx1030 `0.03218873133110086`だった。
- E5 analysis: gfx1030の逆転はscale recipeが第一候補である。E5M2 block16は最大有限値`1.75 * 2^15`を飽和させないため、
  block amaxの仮数が`1.75`を超えるとscaleを標準MXFP8より一段大きくする。標準MXFP8は最大側をSATし得る代わりに残りへ
  2倍細かい刻みを保つ。仮数2 bitではこの差がblock16の局所性を上回ったと推定するが、scale/SAT/layer別countは未取得である。
- decision: freeze済みpolicyのtop-1 `>=0.99`とlong-context loss `<=0.02`を両targetが満たさず、gfx1201／gfx1030は
  旧recipeについて`retain-fp16`。gfx942は`insufficient-evidence`のまま将来のMI300X一括検証へ延期した。明確な品質FAIL後の
  early-stopにより旧recipeの7行performance/resourceを実行しない。
- follow-up: ユーザー決定によりblock16はblock size 16を維持し、descriptorを`kv-fp8-e4-block16-v2`／
  `kv-fp8-e5-block16-v2`、scale recipeを`StandardMxFloorPowerV1`へ変更する。有限amaxの最大2冪をE4では256、
  E5では32768で割ってE8M0範囲へ収め、RNE＋SATする。fresh correctnessはgfx1201／gfx1030でPASSしたが、block16 KLD p99は
  それぞれ`0.006562189165612111`／`0.03659844555378746`、top-1とlong-contextがthreshold未達で両方`retain-fp16`とした。
  default mappingは空、MI300Xは引き続きdeferredである。
- rollback: mapping候補は空で省略時FP16を変更していない。explicit新形式とstate identityは残し、unsupported scopeをsilent fallbackしない。
- evidence: v2 summary `external:phase53/phase53-kv-default-summary-standardmx-v2.json` SHA-256
  `c259e81bc76fb341e9dbba8cdcc0c132456a0762585b471556267cdc59165e10`、空mapping候補
  `external:phase53/phase53-runtime-mapping-standardmx-v2.json` SHA-256
  `283911d387c67d7ba25546ce702fd89ee6a25db482d9e62bf0b61854d5613e77`。個別report digestは
  [Phase 53履歴](../history/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md)を正とする。

### OUT-2026-08-25-P51-GDN-W64: gfx942 wave64 column-state候補（N1・明示opt-in）

- scope: logical target `gfx942`でruntime実体が厳密に`gfx942:sramecc+:xnack-`、Q/K heads 16、value heads 32、
  head/state dim 128、BF16 activation、FP32 recurrent state、token count 128以上。`gfx1030`、`gfx1201`、suffixなし／別suffix、
  unknown target、shape外、127以下は既存providerを維持する。
- baseline/candidate: baselineはvalue head当たり128 threadで各output columnの128 state項を逐次走査する。candidateは
  256 threadの4 wave64でwave当たり1 column、lane当たり2 state dimensionをregisterへ保持し、tokenを逐次処理する。
  recurrent kernelはLDSとbarrierを使用せず、`S^T k`と`S^T q`だけをwave64 treeへ変更する。Q/K L2 normとoutput RMSNormは
  gfx942 baselineと同じ128項index順FP32逐次和、同じBF16-RNE stageを維持する。
- 分類: **N1**。real-number式、128項、FP32 accumulator、decay/state update、transactional state publicationを維持し、
  recurrent projectionの加算依存深さだけを127段からlane local 2項＋wave64 treeへ短縮する。数値差はこの2 reductionへ局所化する。
- selection/identity: `SLLM_LINEAR_ATTENTION_GFX942_WAVE64_COLUMN_STATE=1`の明示opt-inだけで
  `linear_attention.gdn.column_state.gfx942_wave64.v3` / `sllm_linear_attention_column_state_wave64_v3`を選ぶ。
  `SLLM_GDN_FORCE_BASELINE=1`が常に優先する。compile sourceはCMakeがexact `gfx942:sramecc+:xnack-`にだけ設定する
  `SLLM_HIP_COMPILE_WAVE64=1`へ限定する。
- correctness/performance: host/Rust selector、127/128/129境界、shape、target suffix、force-baseline、metadata identityとHIP compileを
  確認した。MI300X operator 7 shapeと、同じstateへのcandidate 128 token→force-baseline 128 token継続はPASSし、second outputを
  256 token逐次scalar oracle、publication length/layoutを128/256境界へ照合した。最大絶対／相対誤差は
  `0.00390625`／`0.014705882`でbaselineと同一、fallbackなし、cleanup 0だった。full-model `10,001/2`は出力完全一致で、
  prefill中央値を`22.718162442`秒から`6.410255551`秒へ3.54403x短縮した。候補入り全7行は未取得のため、default採用せず
  exact gfx942の明示opt-in `target-separated`候補とする。
- rollback: opt-inを設定しないか`SLLM_GDN_FORCE_BASELINE=1`を設定する。恒久rollbackはv3 selectorと3 stage launch/symbolを除去する。

### OUT-2026-08-24-P52: RDNA長context KV providerとVMM append rollback（N0）

- scope: exact `gfx1030`/`gfx1201`のlogical capacity 65,536以上と、共通virtual-contiguous KV append/COW。
  exact `gfx942`の既存resident選択、65,535以下、unknown target、KV layout/encoding、attention kernelは変更しない。
- baseline/candidate: 長capacityのRDNAはpage単位VMM commitからlogical capacity全量の通常device allocationへ変える。
  virtual経路は各planeを逐次更新していたgrow/COWをappend transactionで包み、失敗時に追加mapping/handleと旧shared accessを戻す。
- 分類: **N0**。K/Vのtoken-major byte layout、append入力、attention式、dtype、演算順、丸め、logical publication時点は不変である。
  providerは同じcontiguous pointer contractの物理所有方式だけを変え、失敗rollbackは成功出力を変更しない。
- correctness/output影響: capacity 65,535/65,536/65,537のtarget selector、VMM createのfirst/middle/last、map/access、
  COWのfirst/cross-plane failureを注入し、logical/mapped/commit復元、retry、release、live resource baselineを確認した。
  R9700 `10,001/2`は13/13、`100,000/2`は4/4 PASSし、全requestの生成は`[23066,23066]`、HIP-only、fallback/cleanup 0だった。
- decision/rollback: exact targetとcapacity境界に限定採用する。rollbackは
  `RDNA_CONTIGUOUS_LONG_KV_MIN_TOKENS`のgfx1201選択を除去し、virtual providerへ戻す。VMM transactional rollbackは
  correctness修正なのでprovider selectorのrollbackとは分離する。
- resource: 100kは8 KV layerのK/V 4 GiBをresident確保し、HBM peak `15,388,794,880` bytes、終了後baseline復帰。
- 詳細: [Phase 52計画](../plans/archive/2026/08/21-31/phase52-r9700-100k-kv-commit-oom.md)、
  [summary](../../ci/matrix/phase52-r9700-kv-commit-summary-v1.json)。

### OUT-2026-08-24-P49-P50-BUNDLE: Qwen decode 3融合（N0・target限定採用）

- scope: 固定Qwen3.5-4B BF16 graph、text-only greedy、exact `gfx1030`/`gfx1201`、`M=1`、FP16 KV、adapter/control、
  MTP、multimodal、FP8 sidecarなし。Residual RMSNorm、GDN qkv/z/b/a projection、MLP gate/up/SiLUの3 familyを対象とし、
  `M>1`とscope外graphは既存opへ意味分解する。exact `gfx942`とunknown targetは選択しない。
- baseline/candidate: Residual RMSNormはF32 add→BF16-RNE intermediate→wave32 RMSNorm→BF16-RNEを1 kernelへまとめるが、
  中間roundと8 wave reduction順を維持する。GDNは4本のdecode matmulを一launchへ束ね、各columnのBF16 decode、K項のF32
  accumulate、wave32 tree、BF16-RNEを維持する。MLPはgate/upの独立F32 reductionとBF16-RNEを同じblockで実行し、丸め済み
  gateをSiLU→BF16-RNE、丸め済みupとF32 multiply→BF16-RNEする既存elementwise境界を維持する。
- 分類: 固定Qwenのfinite input/model scopeでは3件とも**N0**。real-number式、入力項、F32 accumulator、wave32 reduction、
  BF16 round stage、公開tensor境界を変更しない。GDN bundleのNaN payload canonicalizationはbaselineとのbit-exact比較を未実施であり、
  非有限payloadをN0へ一般化しない。説明不能なfinite差、state差、非決定差はN3としてcandidateを無効化する。
- correctness/output影響: Residualは`1x2560`、`2x255`、`3x256`、`3x257`のintermediate/output bitwise oracle、
  GDNはactual `K=2560`、width `8192/4096/32/32`を4 baseline matmulへ20 repeat照合した。いずれもgfx1030 evidenceである。
  MLP専用operator oracleと3件のgfx1201専用scalar oracleは未実施だが、gfx1201のcontrol/residual/GDN/MLP/fused3は3 warmup＋
  10 measured、HIP-only、fallbackなし、cleanup復帰、全sampleの生成token列一致をPASSした。5 candidate間でも生成token列は一致した。
- performance/resource: gfx1201 short 17/17のE2E中央値はcontrol `451.8648785` ms、Residual `446.7119175` ms、
  GDN `435.1726845` ms、MLP `436.8484785` ms、3融合 `410.794651` ms。3融合はcontrol比9.09%短縮し、dispatchは
  `108732`から`73372`へ減少した。各runはprocess終了後にHBM/GTT baselineへ復帰した。
- decision/rollback: exact targetごとの専用selectorで限定採用し、対応targetではunset/`1`を有効、`0`/unknownを無効とする。
  rollbackは`SLLM_QWEN_GFX{1030,1201}_RESIDUAL_RMSNORM_FUSION=0`、
  `SLLM_QWEN_GFX{1030,1201}_GDN_PROJECTION_BUNDLE=0`、
  `SLLM_QWEN_GFX{1030,1201}_MLP_GATE_UP_SILU_BUNDLE=0`。詳細は
  [Phase 50履歴](../history/2026/08/21-31/phase50-r9700-port-and-mi300x-handoff.md)を参照する。

### OUT-2026-08-24-P49-P50-P32: decode GQA4 32 partition（N2・target限定採用）

- scope: exact `gfx1030`/`gfx1201`、causal attention decode `M=1`、KV長4,096以上、Q heads 16、KV heads 4、
  head dimension 256、FP16 KV。KV `4095`以下、別head/dtype/target、force-baselineでは既存providerを維持し、gfx942は選択しない。
- baseline/candidate: baselineはkey順の単一online-softmax stateを持つ。P32はKVを32区間へ分け、128-thread/4-waveのstage 1で
  partition-local maximum、denominator、weighted Vを計算し、stage 2がpartition順に固定mergeする。real-number式、causal key集合、
  GQA mapping、F32 state、最終BF16-RNEは同じだが、QKの加算依存深さは概ね8段から12段へ増え、partition merge順も変わる。
- 分類: **N2**。Phase 33 C1およびPhase 49のユーザー承認済みGQA split数値方針と同じく、僅かなworst-case bound増加を
  token一致だけでN0/N1へ再分類しない。gfx1201への展開も同一algorithm/scopeをtarget別A/Bして採用し、別shapeへ拡張しない。
- correctness/output影響: gfx1030はKV `1023/1024/1025/4096/8192/16384`の独立scalar oracle 6/6と、境界・非有限を含む
  full candidate 22/22をPASSし、最大絶対誤差0、fallback/cleanup 0だった。gfx1201はhost selectorで`4095/4096/4097`、env、
  force-baseline、shape、gfx942非選択を確認し、4,096/256 full-modelを1 warmup＋3 measuredでcontrol/P32/P32+3融合とも
  HIP-only、fallbackなし、cleanup復帰、全sample token一致でPASSした。gfx1201専用scalar oracleは未実施であり、その点を隠さない。
- performance/resource: gfx1201 4,096/256のE2E中央値はcontrol `10417.448939` ms、P32 `7671.955934` ms、
  P32+3融合 `6969.896471` ms。P32単体は26.36%、統合候補は33.09%短縮し、統合時decodeは`42.6583` token/sだった。
  P32単体のdispatchはcontrol `504000`から`512160`へ増えたが、全runでHBM/GTTはbaselineへ復帰した。
- decision/rollback: exact target専用envのunset/`1`で既定有効、`0`/unknownで無効とする。
  `SLLM_CAUSAL_ATTENTION_GFX1030_DECODE_GQA4_SPLIT_P32=0`または
  `SLLM_CAUSAL_ATTENTION_GFX1201_DECODE_GQA4_SPLIT_P32=0`でtarget単位に戻し、
  `SLLM_CAUSAL_ATTENTION_FORCE_BASELINE=1`でattention candidate全体をbaselineへ戻す。wave32 block/partitionをgfx942へ直接使わない。

### OUT-2026-08-21-P36-MTP: gfx942 MTP width/state/admission拡張（N0）

- scope: Qwen3.5-4B、公開CLI、exact `gfx942`、greedy MTP draft width 1〜8。BF16 targetはFP16 KV、FP8 targetは
  dynamic FP8 target KVを使い、MTP side modelは既存どおりBF16 weights＋FP16 KVとする。
- baseline: 公開CLIはforced MTPをgfx1201、width 1へ固定し、request stateをvisible token budgetだけで確保していた。
  quantized GGUF plan schemaもMTP graph validationで拒否していた。
- candidate: proposal widthを1〜8へ一般化し、target verify rowsを`width+1`、allocated state capacityを
  `logical+width`へboundedにする。quantized GGUFは同じmodel fingerprint、tied embedding、MTP component/recipeを検証して
  admissionする。target側とMTP側のweight/KV encodingを別fieldでreportする。
- 分類: **N0**。targetが選んだtokenだけを既存順でpublishし、不一致draftはrewind/replayする。target equation、dtype、
  round stage、sampling、stop、visible-token budgetを変えず、追加capacityは未公開の投機stateだけを保持する。
- correctness/output影響: BF16 off/width 2/3/4/7/8とFP8 target off/width 3は、それぞれoffと同じ16 visible tokenへ一致した。
  proposal accountingは全rowでaccepted+rejected=proposed、fallback/cleanup 0、HIP-onlyだった。初回width 2のcapacity overflowと
  FP8 plan schema拒否は修正後のfocused rerunで解消した。
- performance/resource: Session Cはcorrectness runであり性能claimを行わない。追加state slackは最大8 tokenにboundedである。
  rollbackはforced gfx942/width拡張とquantized-plan admissionを除去し、target-onlyまたは従来width 1へ戻す。
- evidence: [Session C summary](../../ci/matrix/phase36-mi300x-session-c-summary-v1.json)。

### OUT-2026-08-21-P36-CHUNK: 公開prefill chunk overrideとMI300X partition確認（N0）

- scope: Qwen3.5 dense公開CLIの`--prefill-chunk-tokens 1..16384`、exact gfx942、BF16 targetのFP16/dynamic FP8 KV。
- change: auto selectorを維持しつつ、明示指定時は一つの候補だけを選び、resource fallbackで別chunkへ黙って変更しない。
  absolute position、KV/GDN state継続、terminal行だけのLM head/Argmax、量子化recipeは既存Phase 31 contractを維持する。
- 分類: **N0**。演算対象、dtype、round stage、terminal visible outputを変えないscheduling/resource指定である。
- correctness/output影響: auto/512/2K/4K/8K/16K × 上記2 KV encodingの12/12 rowで、入力ID`23066`×10,001から
  生成ID`[23066,23066]`へ一致した。全rowはHIP-only、fallbackなし、cleanup 0で、終了後HBM/GTT baselineへ復帰した。
- evidence: [Session B summary](../../ci/matrix/phase36-mi300x-session-b-summary-v1.json)。
- resource: arena high-waterはauto/16K `5,278,049,280` bytes、512 `270,209,024` bytes。これはmemory feasibility
  evidenceであり、single-run timingを性能claimへ使わない。rollbackは明示overrideを除去して既存auto selectorだけへ戻す。

### OUT-2026-08-21-P36-G: gfx942 GDN normのPhase 29 scope修復（N0）

- scope: exact `gfx942` / wave64、Qwen3.5 GDNの短いbaseline provider（token count 128未満）。
  Phase 29で承認したwave32 treeは引き続きexact `gfx1030`/`gfx1201`だけを対象とする。
- baseline: Phase 28までのgfx942はQ/K L2 normとoutput RMSNormを128項のindex順FP32逐次和で計算した。
  Phase 29の共通source変更により、文書化したtarget scope外のgfx942にも4個のwave32 partialを足すtree順が漏れていた。
- candidate: `SLLM_HIP_COMPILE_WAVE64=1`のbuildだけ128項の逐次和を維持し、wave32 buildはPhase 29のtreeを変更しない。
  real-number式、入力集合、dtype、BF16 round stage、recurrent state、kernel symbol、dispatch数、ABIは不変である。
- 分類: **N0**。新しい数値順序の採用ではなく、gfx942を承認済みのPhase 28順序へ戻すtarget-scope修復である。
  gfx1030/gfx1201のPhase 29 N1最適化と、gfx942の既存wave64 matmul providerは変更しない。
- correctness: gfx942のtoken 1/3/17 GDN独立oracleは3/3 PASSし、最大絶対/相対誤差は
  `0.00390625`/`0.014705882`、state publication一致、fallback/cleanup 0だった。Qwen BF16 `Hello`の5-tokenは
  修復前後とも`[11,353,2688,4313,310]`で、同一provider repeatも一致した。診断用wave32 matmul controlではGDN順序との
  組合せだけがreviewed RDNA token `[11,353,1044,4313,310]`への分岐を説明した。
- output影響: final gfx942 BF16とFNUZ FP8は3番目のtokenだけ`2688`/`1044`へ分岐した。BF16はwave64 BF16 reduction、
  FP8はhipBLASLt FNUZを使うため、このcross-dtype/cross-provider差をbit-exact gateにはしない。Unicodeとstop rowは一致した。
- performance/resource: 未測定。短GDNのdispatch、scratch、allocationを変更せず、Phase 29/35のRDNA performance scopeへ
  影響しない。rollbackはwave64条件分岐の除去だが、Phase 29の承認scopeを再びgfx942へ拡張するため採用しない。

### OUT-2026-08-20-P35-A: Full Attention Q_TILE=4 query-row共有（N1・限定採用）

- scope: exact `gfx1030`/`gfx1201`、Qwen系causal/full attention、`M>=128`、Q heads 16、KV heads 4、
  head dim 256、FP16/dynamic FP8/static FP8/NVFP4 KV。短M、decode、別shape/targetは既存providerを維持する。
- baseline: Phase 33 C2が1 query row × 1 KV head/workgroupでK/VをGQA 4 headへ共有する。
- candidate: 1 workgroupが4 query row × GQA 4 headを所有し、K/Vを16 logical queryへ共有する。各logical queryの
  causal key集合、key順online softmax、FP32 maximum/denominator/weighted V、BF16 RNE出力は独立に維持する。
- 分類: **N1**。QKは同じ256項を8 value/laneの固定treeとwave32 treeで加算し、Phase 33 C2の概ね8段を超えない。
  real-number式、入力集合、dtype、丸めstageは同じで、標準worst-case boundは非増加である。
- correctness: 2 target × 4 KV encoding × 29 caseの232/232 PASS。M=127/128/129、255/256/257、nonzero start、
  long-prefix decode、NaN/+Inf/subnormalを含み、最大絶対誤差はFP16 `2.3841858e-7`、FP8 `4.7683716e-7`、
  NVFP4 `1.1641532e-9`、fallback/cleanup 0だった。
- output影響: fixed 10,001 input / 2 outputはbaseline/candidateとも`[2064,5686]`。将来の差は同じ項の固定tree順へ
  局所化できる範囲だけN1とし、説明不能・非決定差はN3とする。
- 性能/resource: V620 profileのFull Attentionは10.820秒から4.110秒へ62.02%短縮し、Attention-only E2Eは
  V620 19.31%、R9700 8.59%短縮。global scratch、追加dispatch、KV mirrorは0、arena high-waterは不変。
- 決定: 担当AI裁量で両target共通のshape限定採用。M=64/65のV620候補悪化を避けるため境界を128とし、
  Phase 33 providerを明示complementにする。固定改善率は使用しない。
- rollback: `SLLM_CAUSAL_ATTENTION_FORCE_BASELINE=1`相当のselectionへ戻し、Q_TILE=4 symbol/launchを除去する。
- 詳細: [Phase 35履歴](../history/2026/08/11-20/phase35-long-context-full-attention-gdn-optimization.md)、
  [bounded summary](../../ci/matrix/phase35-attention-gdn-summary-v1.json)。

### OUT-2026-08-20-P35-G: GDN column-owned recurrent state（N1・限定採用）

- scope: exact `gfx1030`/`gfx1201`、Qwen3.5 GDN、token count 128以上、Q/K heads 16、value heads 32、
  head/state dim 128、BF16 activation、FP32 recurrent state。短prefill/decodeはPhase 28/29 providerを維持する。
- baseline: value head当たり1 workgroupで、threadが1 output columnを所有し、128 state rowを逐次走査する。
- candidate: preprocessでQ/K normとbeta/decayを一度生成し、1,024 workgroup相当のcolumn-owned recurrent kernelで
  lane当たり4 state rowをregisterに保持し、postprocessで既存output RMSNorm/z SiLUを適用する。targetごとの既存物理state
  index mappingとtransactional previous/next publicationは変えない。
- 分類: **N1**。state transpose/migrationや項の欠落はない。`S^T k`/`S^T q`の同じ128 FP32項を逐次依存127から
  4項local + wave32 treeの概ね8段へ短縮し、標準worst-case boundは非増加である。Q/K、beta、raw output、normalized outputの
  BF16 round stage、decay/state update式は維持する。
- correctness: 両targetでtoken 1/3/17/127/128/129を独立oracleへ照合し12/12 PASS。最大絶対/相対誤差は
  `0.00390625`/`0.014705882`、next-state publication一致、fallback/cleanup 0だった。
- output影響: fixed 10,001 input / 2 outputは`[2064,5686]`を維持した。将来の差はrecurrent projectionの固定tree順へ
  局所化できる範囲だけN1とし、state/publication差はcorrectness blockerとする。
- 性能/resource: V620 GDN familyは約7.672秒から0.618秒へ91.95%短縮しfixed llama.cpp 0.622秒と概ね同等になった。
  GDN-only E2EはV620 19.84%、R9700 7.17%短縮。10,001 tokenでbeta/decay FP32 planeを2,560,256 byte/layer、
  24 layer合計61,446,144 byte追加し、
  1 layer当たりdispatchは2から4、full-modelは984から1,032へ増えたがarena high-waterは不変だった。
- 決定: 担当AI裁量で両target共通のshape限定採用。絶対短縮、N1、peer parity、既存state layout再利用、短経路complementを
  総合し、追加2 dispatchの費用を上回ると判断した。
- rollback: `SLLM_GDN_FORCE_BASELINE=1`相当のselectionへ戻し、preprocess/recurrent/postprocessの3 candidate kernelを除去する。
- 詳細: [Phase 35履歴](../history/2026/08/11-20/phase35-long-context-full-attention-gdn-optimization.md)、
  [bounded summary](../../ci/matrix/phase35-attention-gdn-summary-v1.json)。

### OUT-2026-08-20-P34: gfx1030長行BF16 matmul hipBLAS route（N1・限定採用）

- scope: exact `gfx1030`、Qwen3.5-4B内部BF16 projection。主要5 shapeは`M>=128`、`K=2560,N=1024`は
  `M>=1024`。N=32、未知shape、all-logits、短M、gfx1201/gfx942は既存providerを維持する。
- baseline: `matmul.bf16_fp32.tiled16.v2`が16x16 tile内でKをsource-level scalar FP32 accumulateし、BF16 RNE出力する。
- candidate: existing `matmul.hipblas.gemm_ex.v2`/`hipblasGemmEx`を使う。BF16 input/weight、FP32 compute、BF16 RNE出力、
  real-number equation、入力項集合、layout、publicationは同じ。観測Tensile solutionはGSU1でglobal split/atomic combineを使わない。
- 分類: **N1**。providerの固定reduction順は異なるためbit exactのN0ではないが、同じK項を一度ずつ含む決定的な並べ替えであり、
  Phase 8の保守的な`gamma_K * sum(abs(a_i*w_i)) + BF16 half-ULP` worst-case boundは非増加である。
- correctness: signed/exponent-mixed stressはprovider間差を観測したが、repeatは決定的だった。`M=128,K=2560,N=4096`と
  `M=10001,K=2560,N=9216`のsampled F64 bound違反は両provider 0、matmul G1は両target 18/18 PASS、fallback/cleanup 0。
- output影響: final 10,001 prompt / 2 outputはbaseline/candidateともtoken `[2064,5686]`。別入力でtoken差が生じても、
  同じ式・項・dtype・丸めstageと非増加boundへ局所化できる範囲はN1として扱う。説明不能な差はN3である。
- 性能/resource: V620 248-call加重projectionを62.526秒から11.081秒へ82.28%、full modelを89.249秒から
  34.684秒へ61.14%短縮。context-lifetime hipBLAS handle 1、hipBLASLt/workspace/weight repack/追加dispatch 0、arena不変。
- 決定: 担当AI裁量でshape-aware限定採用。small-Nの不安定性と未知shapeは既存providerへ隔離し、固定改善率は使用しない。
- rollback: `phase34_gfx1030_hipblas_shape` routeとgfx1030 contextのhipBLAS handle作成条件を除去してtiled16へ戻す。
- 詳細: [Phase 34履歴](../history/2026/08/11-20/phase34-v620-long-prefill-bf16-matmul-provider-optimization.md)、
  [bounded summary](../../ci/matrix/phase34-v620-prefill-matmul-summary-v1.json)。

### OUT-2026-08-20-P33-C1: decode wave8 KV split（N2・ユーザー承認採用）

- scope: exact `gfx1030`/`gfx1201`、Qwen系causal/full attention、`M=1`、KV長1,024以上、head dim 256、
  FP16/dynamic FP8/static FP8/NVFP4 KV。2026-08-20のユーザー承認によりproductionへ限定採用する。
- baseline: 256 dimensionのQK積を概ね8段のbalanced treeで加算し、key順に一つのonline-softmax stateとweighted Vを更新する。
- candidate: 8 waveへ連続したKV区間を割り当て、waveごとのpartial online-softmaxを区間順にLDS上で固定mergeする。
  各laneが8 dimensionを逐次加算してwave treeへ渡すためQK加算依存深さは概ね12段となる。real-number semantic、入力集合、
  FP32 accumulator、softmax式、BF16 RNE、state publicationは維持する。
- 分類: **N2**。QK sumの標準worst-case boundは概略`gamma_8`から`gamma_12`へ僅かに増える。weighted Vは短い
  partialと固定mergeになるが、QK側のbound悪化を相殺したと証明しない。承認後もN1へ再分類せずN2として追跡する。
- correctness: 4 KV encoding × 2 target × 29 caseを含むPhase 33 oracle 232/232 PASS。candidate範囲のKV=1,024
  NaN query/+Inf value、KV=4,097 signed mixed、KV=8,193を含む。最大絶対誤差はFP16 `2.3841858e-7`、FP8
  `4.7683716e-7`、NVFP4 `1.1641532e-9`。fallback/cleanup 0、測定範囲の生成token差なし。
- 性能: KV=1,024〜8,193のdevice中央値をgfx1201で約53〜58%、gfx1030で約64〜65%短縮した。scratch、追加dispatchは0。
- 決定: **ユーザー承認により採用**。大幅なdevice短縮、観測誤差、token一致、scratch/追加dispatch 0を確認したうえで、
  N2の僅かなworst-case bound増加を受容した。C1 symbol/routingをproductionに維持する。
- rollback: `use_decode_wave_split`をfalseとし、`causal_attention_decode_wave_split_kernel`とC1 metadata symbolを除去する。
- 詳細: [Phase 33履歴](../history/2026/08/11-20/phase33-full-attention-structural-optimization.md)、
  [bounded summary](../../ci/matrix/phase33-full-attention-summary-v1.json)。

### OUT-2026-08-20-P33-C2: prefill GQA4 K/V共有

- scope: exact `gfx1030`/`gfx1201`、Qwen系causal/full attention、`M>=64`、GQA ratio 4、head dim 256、
  FP16/dynamic FP8/static FP8/NVFP4 KV。
- baseline: query row/query headごとに1 blockを起動し、同じKV headへmapされる4 query headがK/Vを別々にdecode/readする。
- candidate: 1 query row/KV headごとに1 blockを起動し、K/V elementを一度だけdecodeして4 query headで共有する。各headの
  QK reduction、online maximum/denominator、weighted V、causal key順は独立に維持する。global scratch、追加launch、KV mirrorはない。
- 分類: exact `gfx1201`は既存wave providerと同じ32-lane partial + 8 partial固定treeで**N0**。exact `gfx1030`は
  256-thread LDS treeからwave32 + 8 partial treeへ順序が変わるが、加算依存深さは同じ8段で標準worst-case boundが
  非増加のため**N1**。real-number equation、入力集合、dtype、丸めstageは維持する。
- correctness: Phase 33 oracle 232/232 PASS。M=63/64/65、127/128/129、255/256/257、nonzero start、M=64の
  NaN query/+Inf valueを含む。fallback/cleanup 0、測定範囲の生成token差なし。
- 性能: M=64〜257のFP16 device中央値をgfx1201で約21〜47%、gfx1030で約38〜54%短縮した。M=37のgfx1201
  prototypeは8.22%悪化したためM>=64だけへstable routeする。R9700 10,000-promptのC1+C2候補はB0比28.96%全体短縮。
- 決定: 担当AI裁量でshared限定採用。全scoped patternの大幅改善、scratch 0、共通source、明示B0 complement、低い保守費用を
  総合し、固定改善率gateは用いない。gfx1201 matrix innerは同じ4-row tileへ適合せず別candidateとして棄却する。
- rollback: `use_prefill_gqa4`をfalseとし、M>=64をPhase 30 wave provider（gfx1201）またはB0（gfx1030）へ戻す。
- 詳細: [Phase 33履歴](../history/2026/08/11-20/phase33-full-attention-structural-optimization.md)、
  [bounded summary](../../ci/matrix/phase33-full-attention-summary-v1.json)。

### OUT-2026-08-19-P32: gfx1201 native FP8 KV append encode

- scope: exact `gfx1201`のdynamic/static FP8 KV append。gfx1030、FP16、NVFP4は既存経路を維持する。
- baseline: scale後のF32値をsoftware binary searchでOCP E4M3FNへRNE/saturation encodeする。
- candidate: NaN、Inf、signed zero、448 saturationを同じcontractへ明示補正し、通常finite値を
  `__builtin_amdgcn_cvt_pk_fp8_f32(value, value, 0, false)`でencodeする。kernel、workgroup、grid、scale、store、KV formatは不変。
- 分類: **N0**。全65,536 BF16 codeをK/Vで一巡したdynamic/static fixtureと19 token境界でpayload byte／F32 scale bit mismatch 0。
  production attention oracleもgfx1201/gfx1030 × dynamic/static FP8の68/68 caseをPASSした。
- output影響: 測定上もcontract上も変更なし。生成tokenはgfx1201 10,001/16,385、gfx1030 10,001 inputですべて`[1228, 1228]`。
- 決定: 担当AI裁量でC1 native scalarを限定採用。C2 packedはworkgroup/store/tail複雑性のため不採用。default KVはFP16のまま。
- rollback: `float_to_e4m3fn_fp8_append` callをsoftware `float_to_e4m3fn`へ戻す。public ABI、state migration、artifact変換は不要。
- 詳細: [Phase 32履歴](../history/2026/08/11-20/phase32-native-fp8-kv-append-revalidation.md)、
  [bounded summary](../../ci/matrix/phase32-native-fp8-append-summary-v1.json)。

### OUT-2026-08-19-P31: chunked prefillとworkspace arena

- scope: Qwen3.5 dense BF16 weight graph、text prefill、FP16/dynamic FP8/static FP8/NVFP4 KV、gfx1030/gfx1201。
- baseline: prompt全行を一graph実行し、request-owned dynamic tensorを個別bufferへ割り当てる。
- candidate: promptを連続chunkへ分割し、absolute positionとKV/GDN stateを継続する。中間chunkの未使用LM head/Argmaxを省略し、
  completion boundaryまで重なるtensorは別slotのまま、重ならないlifetimeだけを再利用する。
- 分類: **N0**。terminal行のreal-number equation、dtype、演算順、量子化recipe、丸めstageを変更しない。chunk境界で既存KVを
  再量子化せず、新規K/Vを一度だけappendする。static FP8のscale 1.0は明示設定のdescriptor完成でありdefault変更ではない。
- correctness: 10,001 tokenをgfx1030/gfx1201、16,385 tokenをgfx1201でHIP-only、fallbackなし、cleanup 0で実行した。
  16,385 tokenは16,384+1の2 chunkとなり、反復入力の生成tokenはone-chunk controlと同じ1228だった。dynamic FP8は両targetの
  10,001 tokenとgfx1201の16,385 token、static FP8はgfx1201の10,001 token、NVFP4は513-token spotをPASSした。
- output影響: 測定範囲でtoken差なし。chunk partition、arena reuse、intermediate terminal省略から説明できない差はN3 blockerとする。
- resource: 10,001 tokenのworkspace high-waterは39,950,821,120から5,278,049,280 byte、16,385 tokenでは
  65,448,547,584から8,646,688,768 byte相当となり、いずれも約86.79%削減した。
- 決定: shared chunked prefill/arenaを採用し、明示low-bit KV選択をCLI/serverへ接続する。defaultはFP16を維持する。
- rollback: source base commit `1def2b63cfb26cd71e7e1bf500235a6eb5c7ed9b`の一括prefill・個別allocation。
- 詳細: [Phase 31履歴](../history/2026/08/11-20/phase31-chunked-prefill-memory-foundation.md)、
  [bounded summary](../../ci/matrix/phase31-chunked-prefill-summary-v1.json)。

### OUT-2026-08-19-P30: RDNA4 causal-attention wave reduction

- scope: Qwen系generic causal/full attention、exact gfx1201、head dim 256、decode `M=1`とprefill `M>=32`、FP16/FP8/NVFP4 KV。
- baseline: 256 threadのLDS treeでQKの256項FP32積を固定balanced reductionし、keyごとに約11回のblock同期を行う。
- candidate: 8 wave × 32 laneの`__shfl_down` treeと8個のLDS partialを固定treeで合成する。online-softmaxのmax、denominator、V accumulation、FP32 accumulator、BF16 RNE output stageは維持する。
- 分類: **N1**。real-number equation、入力集合、dtype、丸めstageは同じで、QKの加算依存深さはbaselineの8段からwave内5段+wave間3段の8段を超えない。native E4M3FN readは全256 code（NaN 2 codeを含む）がsoftware contractと一致するためN0である。
- correctness: gfx1201/gfx1030 × FP16/FP8の各17 caseが全出力一致、fallback 0、cleanup failure 0。gfx1201 native decode probeは256/256 code PASS。full-model 29/267/4108 inputのbaseline/candidate token recordは一致した。
- output影響: 測定範囲ではtoken差なし。演算順が変わるため将来のlogit/token差は生じ得るが、原因は固定balanced tree間の丸め差へ局所化できる。
- 性能: gfx1201 operatorはFP16 decodeで6.12〜17.16%、FP8 decodeで0.64〜27.91%、prefill `M≈255`で約21.0%/31.5%短縮。Qwen3.5-4B BF16、4108 inputの3 process中央値はTTFT 9.60%、prefill 9.72%、E2E 9.16%、decode throughput 7.86%改善した。29 inputのTTFT -0.60%は1 processのsub-1% control noiseとして非悪化扱いとした。
- 決定: exact gfx1201の`M=1`と`M>=32`へ限定採用し、`M=2..31`とgfx1030はbaselineを維持する。native append encodeはchunk 256で68.69%悪化したため棄却した。
- candidate source SHA-256: closeout summaryの`kernel_source_sha256`を正とする。
- rollback: source base commit `1def2b63cfb26cd71e7e1bf500235a6eb5c7ed9b`のscalar/vector provider。
- 詳細: [Phase 30履歴](../history/2026/08/11-20/phase30-rdna4-native-attention-kv-optimization.md)、
  [bounded summary](../../ci/matrix/phase30-rdna4-attention-kv-summary-v1.json)。

### OUT-2026-08-18-P29: GDN norm wave32 tree reduction

- scope: Qwen3.5-4B dense BF16、通常target-only decode、GDN recurrent gated norm、gfx1030/gfx1201。
- baseline: Q/K L2 normとoutput RMSNormの128項FP32逐次和。依存深さ127。
- candidate: 4 wave × 32 laneの`__shfl_down` tree reduction後、4 partialを固定順で加算。最長加算依存深さは概ね8。
- 分類: **N1**。対象項は二乗値で全て非負、real-number sumとBF16 RNE stageは同じで、逐次和の概略誤差bound
  `gamma_127`をtreeの概略`gamma_8`へ縮小する。race、atomic、追加launch、scratchはない。
- correctness: model-free token 1/3/17は両GPU PASS、output 16の全formal runはtoken一致、同一provider repeatは一致、fallbackなし、cleanup 0。
- output影響: output 128はgfx1030/B0が105、B2が20、gfx1201/B0/B1/B2が111/112/108 token目からbaselineと分岐。
  gfx1030/B1は128 token一致。差はreduction順変更によるrecurrent stateへの丸め差蓄積として説明可能。
- 性能: GDN device p50はgfx1030で2.15〜2.20%、gfx1201で8.10〜9.21%短縮。全pattern非悪化、gfx1201の全patternが5%以上。
- 決定: 2026-08-18の改訂規則でN1としてshared adoption。token完全一致を理由に棄却しない。target splitなし。
- candidate source SHA-256: `62b5d6caab9e06044e29c5f043046a65eace243faafa034aca1b3f2ce8eb3dc6`。
- rollback: Phase 28 source SHA-256 `44e0b6a0b3e5bb01cd21423a2796f14dd20d8a40f5bbc7ac5c56535a38e3807f`。
- 詳細: [Phase 29履歴](../history/2026/08/11-20/phase29-gdn-useful-workgroup-parallelization.md)、
  [bounded summary](../../ci/matrix/phase29-gdn-device-summary-v1.json)。

### OUT-2026-08-18-P28: GDN state pass統合

- scope: Qwen3.5 GDN recurrent state、gfx1030/gfx1201。
- change: 初回state copy、decay、previous projectionを一passへ統合。key走査、FP32演算順、BF16 RNE、state publicationを維持。
- 分類: **N0**。代表output 128でbaseline/candidate token record一致、tiny oracle PASS。
- 決定: shared adoption。Phase 29のrollback provider。
- 詳細: [Phase 28履歴](../history/2026/08/11-20/phase28-decode-nonprojection-device-optimization.md)。

### OUT-2026-08-18-P24: prefill terminal-row projection

- scope: Qwen3.5 prefill terminal LM head/Argmax。
- change: 生成開始に不要な非terminal rowのLM head/Argmaxを省略し、terminal rowの式とproviderを維持。
- 分類: **N0**。生成に使用するterminal logits/tokenを維持し、all-logits要求とMTP経路は既存all-rowへrouteする。
- 決定: shared adoption。
- 詳細: [Phase 24履歴](../history/2026/08/11-20/phase24-prefill-terminal-row-projection-optimization.md)。

### OUT-2026-09-10-P835-R36-R37: attention候補の限定統合と棄却

- R36はN0候補。gfx1201・M2048・既存prefix>=1024のQ8/W16でK/Vをwave内registerへ読み込み、QK reduction／key順／softmax／BF16丸めを維持する。scratchは独立oracle最大1 BF16 ULP、finite pair差0、repeat／非finite分類をPASS。対象3 prefixで約1.6〜4.3%短縮し限定統合した。両target compileとgfx1201統合後4case oracle／repeat／分類検査をPASS。通常モデル速度の確認は残る。
- R37はdenominator／value rescaleの明示FMA化によるN1候補。標準集約誤差bound非増加の解析とattention数値検査は通過したが、測定全shapeで約1〜2%遅く不採用。小さいrecurrence probeはloader errorで未実行であり、GPU PASSではない。
- R32 support captureはN0。固定K20の既存分布を256 Bのprivate workspaceへ保持し、公開16 B選択結果を維持する。V620本体focused6caseをPASS。後続のp/q sampling変更はまだ通常生成へ接続しておらず、このN0判定に含めない。
- 詳細: [Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。full-model BF16品質同等性を認定しない。

## 運用

- 今後、出力へ影響しうる変更はPhase historyだけで完結させず、この台帳へ一項目を追加する。
- token差がない変更も、数値順序、近似、quantization、state、sampling、stopへ触れる場合はN0として記録する。
- 台帳更新は実装と同じcommitに含める。raw model、full logits、生成全文は追跡せず、bounded aggregateとidentityだけを残す。

[メイン計画](../plans/main-plan.md)


### OUT-2026-09-10-P835-R32-PQ: 固定samplingとGPU stochastic MTP

- scope: Qwen3.8 NVFP4、MXFP8 KV、固定T1/P.95/K20、対応する同一sessionのtarget／BF16 companion。supportは256 B、decisionは144 Bで、全語彙やsupportのCPU転送をしない。
- change: draftを独立乱数のqから選び、min(1,p/q)で採用し、棄却時は正のp-q、全採用時はtarget bonusから選択する。targetの固定sampling分布を維持する方式変更として扱い、N1丸め誤差改善と混同しない。非MTP selectorの乱数契約は維持する。
- correctness: production private wrapperを通したgfx1030／gfx1201各35caseを独立dense long-double oracleで確認。正常10、不正record15、wrapper引数10、width1/2/3/8、解放を含む。V初回testのallocation alignment失敗はdispatch前の失敗として保持し、testのみ修正後のarchive再利用対応を保存した。
- output: 8192入力／128出力・seed123の旧R31b MTP→新R32 p/q比較はV7/128、R2/128が同位置一致、両方2token目から分岐。新旧sampling乱数・検証方式の変更を含み、同一binary MTP有無の比較ではない。通常生成は各53/53・50/50 blockでp/qへ到達しHIP-only／fallbackなし／cleanup0。BF16 full-model品質同等性は未証明で、token一致率から品質劣化率を推定しない。
- evidence: `.local-artifacts/phase83-5/pq-public-gfx1030-r32-r2/`、`pq-public-gfx1201-r32-r2/`、`r32-pq-v620a-ordinary-r1/`、`r32-pq-r9700-ordinary-r1/`。詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

- 同一R32 binaryのV620 MTP off/on単回を追加確認し、同位置7/128・共通prefix1。offは従来hash `e38500f139e36819d6ed273c95f17e9384aba6e623106b162b679535efb2d9c9`、onは`316dc877624f4f03edafe20ff3002ef53dd4f4d90742441bcf23dd61e9ba27ff`。model／fixture／seed／chunk2048一致を保存した。これは出力差の再現で、品質同等性は未証明。`r32-v620-mtp-off-on-comparison.json`。

- R9700も同一R32 binaryのMTP off/onを追加確認した。offは従来hash `ded3d447d4d100eaff932a5c70e8be1e55b84af4bfee6649acde91b6dd60c506`、onは`7d939adfc7c9b60bb30677bd1c56d462cc6a3b1b86c7512c29913f5d0bb99155`で、同位置2/128・共通prefix1。`r32-r9700-mtp-off-on-comparison.json`。単回の出力比較でありBF16品質評価ではない。

### OUT-2026-09-10-P835-R39: target support-only readback

- change: p/qが使用しないtargetの公開16 B selector recordだけをCPUへ転送せず、GPU support／sampling式／乱数／state操作を維持するN0変更。companionの抽選結果とsupport D2Dは維持する。
- failure: 初回Vモデルは公開token数をstate出力行数としていたsliceで失敗した。入力blockの実行数を渡すよう修正し、公開recordなしの部分／全採用slice回帰を含む6testをPASS。初回失敗を速度／GPU PASSとはしない。
- output: 修正r2のV620通常8192/128でR32と128token、p/q53block、draft74/106採用、dispatch数が一致し、HIP-only／fallbackなし／cleanup0。両target build／source identityと無変更native archiveの対応を保存。R9700と通常APIは確認中。詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

- R39 r2 R9700も128token・p/q50block・draft78/100採用がR32と一致し、HIP-only／fallbackなし／cleanup0。V APIの5caseもPASSし、MTP提案2／採用2後のcancel、復旧、長入力text hash一致、解放を確認した。decode単回はR32 24.360→R39 23.929 tok/sであり、この差を速度改善とは認定しない。


### OUT-2026-09-10-P835-R40: MTP state-onlyのKV後の不要計算削除

- scope: 単一full-attention companionの明示state-only呼出し。prefix primingと全採用後の状態合わせで使う。通常targetやMTPの出力を必要とする呼出しは全tailを維持する。
- change: Q/K/V／preprocess／KV encodeとpublicationを維持し、その後の読まれないattention／MLP等を省くN0変更。後続のstate依存がないことをgraphで確認し、未消費appendや空segmentの誤完了を拒否する。
- correctness: host M1／容量超過前の無変更／次の通常draftと関連29testをPASS（既存3 ignored）。実8192/128では両GPUでR39と128token・MTP採否が一致、target dispatch不変、companion dispatchはV507／R543減少。HIP-only／fallbackなし／cleanup0。host容量1fixtureをM2/M3の証拠にしない。
- evidence: `.local-artifacts/phase83-5/r40-ordinary-comparison.json`、`r40-native-evidence-reuse.json`。単回速度はV219.257／25.143、R447.964／24.375 tok/sでR目標未達。BF16 full-model品質同等性の追加主張はない。詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-10-P835-R43: MTP入力のBF16行結合

- classification: N0。連続BF16行列2個を列方向へ結合するuint16 load/storeのみで、演算・丸め・samplingを変更しない。nativeはモデル寸法に依存せず、実M>1のMTP入力へ接続する。
- correctness: V620 exact gfx1030のfresh archiveでM1/2/3/127/128/129、非整列列数、offset、guard、signed zero／subnormal／NaN payloadのbitwise oracleとread-only input alias、エラー境界をPASS。core側は連続stride・範囲・overflow・output overlapを検査する。R9700と通常モデルの確認は進行中。
- evidence: `row-concat-public-gfx1030-r43-r1`。出力品質のBF16比較や通常モデル速度の証拠へは拡張しない。詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

- V620通常8192/128でもR40と128token・MTP採否が一致し、HIP-only／fallbackなし／cleanup0を確認した。`r43-v620-output-comparison.json`。

- R9700通常8192/128もR40と128token・MTP採否が一致、HIP-only／fallbackなし／cleanup0。`r43-r9700-output-comparison.json`。

- V620 R43 APIも5case PASS、8192/128 text hashはbenchmarkと一致。MTP進行後cancelと復旧、cleanup0を確認。BF16品質同等性の証拠ではない。

- R9700 fresh archiveのconcat単体もbitwise／境界PASS。両GPUの単体と通常生成の検査を完了し、R43は採用候補として維持する。

### OUT-2026-09-10-P835-R45-R47: metadata取得と既存attentionの接続修正

- classification: N0。R45はmetadataの取得を既存native直接queryへ変更し、前後のstate length・identity・generation・physical metadataのチェックを維持する。readbackのlive viewは維持する。R47はR36のhost-side device macro誤用をruntime GPU／M条件へ置き換える。数値式・型・K/V読込み後の計算順を変更しない。
- R36の過去の統合数値PASSだけではWaveLocalKv=trueへの到達を証明できていなかった。R47 fresh archiveでM2048の3prefixとM129境界のoracle／全出力比較をPASSし、通常profileのtrue48回・false0回を確認した。
- 両GPUの通常8192/128はR43と128token・MTP採否・dispatch数が一致、HIP-only／fallbackなし／cleanup0。BF16 full-model品質の未測定範囲は変わらない。詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。


### OUT-2026-09-10-P835-R48: FP8 operand LDSのrow padding

- classification: N0候補。NVFP4→E4M3FN exact ingress、scale値、K16 WMMA／FP32 Kahan／BF16丸め順を維持し、LDSの物理row strideだけ64→68 Bに変更する。通常採用予定範囲はgfx1201・M2048・K5120/N17408と逆tuple。
- evidence: `r48-r52-r9700-scratch-r1/root-audit.json`。実aligned controlに対するfinite全出力pair差0、repeat、各64sampleの独立long-double oracle最大0 ULP、candidate自身のNaN fixture分類をPASS。M128/127/129はcontrolのみを実行し、candidateの証拠へ拡張しない。
- status: 本体へ接続済み。R48+R52のfresh公開gfx1201経路でM255/256/257/2047/2048/2049・両tupleの12caseをPASSし、M2048のpadding選択と隣接shapeの既存選択を確認した。両GPUの通常8192/128でR47と128token・MTP採否が一致し、HIP-only／cleanup0。scratchの約20.7〜21.0%時間短縮とfull-model速度は区別する。`r48-r52-prefill-public-gfx1201-r1`、`r48-r52-v620-output-comparison.json`、`r48-r52-r9700-output-comparison.json`。BF16 full-model品質同等性の追加認定はない。詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。


### OUT-2026-09-10-P835-R52: E2M1 byte展開の整数命令削減

- classification: N0。4項のmasked shiftを2段のbit spreadへ置換し、E4M3FN lookupとsignを維持する。全65,536入力のhost／gfx1201 codec oracle、65535/65536/65537境界・guard・repeatをPASS。
- evidence: `r52-r9700-matched-r1`。M256/1024/2048・2 tupleのaligned candidate8caseでfinite全出力pair差0、NaN分類／repeat一致、sampled long-double oracle最大0 ULP。M257はcontrol-only。scratchで約1.3〜1.6%高速だったが、他shapeやfull-model速度へは一般化しない。
- status: R48と合わせて本体へ反映済み。上記R48のfresh公開GPU12caseと両GPU通常モデルで数値・出力・解放を確認した。V620正式1 warmup＋3 measuredでも全128token一致。R9700単回はprefill517.648／decode24.720 tok/sでdecode目標未達のため、Phase83.5は継続中。BF16 full-model品質同等性を追加認定しない。詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-10-P835-R53: M2〜4のNVFP4 gate/up入力量子化共有

- classification: N0候補。既存projection packの共有対象を実M2/3/4へ広げ、同一BF16入力とinput-global scaleの量子化を1回にまとめる。両memberは既存ID94、gate→up順、量子化byte／scaleと丸めを維持する。M1とM5〜63の既存経路は維持する。
- scope: exact gfx1030／gfx1201、検証済みQwen3.8 targetの既存NVFP4 gate/up pair。MTP companion自体のBF16演算は変更しない。graph容量ではなく実token_countで選び、Unsupportedのfallbackはprepare時に限定する。
- status: core関連8test、HIP wrapper境界1test、CPU-only公開runtime host testと両targetのfresh buildをPASS。続いて両GPUの公開経路でM2/3/4の行別入力による全出力oracle、direct/shared bitwise一致、repeat、workspace／dispatch、M5 Unsupportedと解放をPASSした。通常8192/128も両GPUで128token・MTP採否がR48+R52と一致し、target dispatchのみV2,968／R2,800減、HIP-only／cleanup0を確認した。V単回218.998／25.117、R515.377／24.923 tok/sで、Rのdecode目標は未達。証拠は`r53-gfx1030-output-comparison.json`／`r53-gfx1201-output-comparison.json`、`r53-verification/`、`r53-pack-public-gfx1030-r1`／`r53-pack-public-gfx1201-r1`、詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-10-P835-R55: 既存ResidualRmsNorm融合の適用拡大

- classification: 有限入力についてN0候補。residual addのFP32和を従来どおりBF16へ丸め、その中間値を使って同じwave reduction順・epsilon・OffsetOne scale・BF16出力丸めを計算する既存kernelを再利用する。中間residual tensorも出力として保持する。
- scope: 検証済みQwen3.8 artifactとrecipe digestが一致する64-layer targetの隣接64 pair。既存Qwen3.5-4B scopeは維持し、MTP companionは対象外。native kernel／公開ABIは追加しない。llama.cppはQwen35 graphのresidual／FFN接続を構造参照したもので、source importは行わない。
- status: V620のH5120 M1/2/3・Direct／OffsetOneについて、高精度CPU数学oracle・BF16 Add中間・通常GPU RMSNormとのbitwise比較をPASSした（`r55-rmsnorm-public-gfx1030-r2`）。初回r1はCPU逐次FP32和の誤差でOffsetOneのBF16中点を跨いだためFAILし、独立再現とGPU pair診断後に新H5120 oracleだけlong doubleへ修正した。数値kernel／許容差は変更せず失敗記録も保持した。R9700公開r2もPASSし、core統合後の両GPU通常8192/128で出力・採否がR53と一致、HIP-only／cleanup0を確認した。target dispatchはV3,648／R3,456減ったが、R9700単回decode24.790はR53の24.923より遅く、速度改善は未確認。非有限値のpayloadやBF16 full-model品質の未測定範囲を認定しない。調査は`r55-qwen38-residual-rmsnorm-expansion.md`、詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

### OUT-2026-09-10-P835-R54: MTP companion hidden行のGPU内引渡し

- classification: N0候補。fixed-K20 p/qのcompanion proposal間で、同じBF16 hidden行のhost readback／uploadを同一queueのD2Dへ置換する。最初のtarget hidden uploadは保持し、最終proposalの未消費hiddenはreadbackしない。数値kernel・RNG・採否・commit／rollbackの演算は変更しない。
- scope: 現行fixed-K20 p/q MTP経路。その他selector経路は従来host hiddenを保持する。native ABI変更なし。
- status: host recorderでwidth1/2/3の転送契約とcopy／graph失敗を検査。両GPU通常8192/128でR55との128token／text hash、MTP採否・dispatch一致、HIP-only／cleanup0を確認した。転送数はhost recorderの観測と実GPU traceを区別する。V620単回decode25.073→22.077へ低下したため速度採用は保留、R9700単回24.850も目標未達。証拠は`r54-gfx1030-output-comparison.json`／`r54-gfx1201-output-comparison.json`、詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。

- R54最終判断: 速度改善を確認できず、本体から専用経路を削除した。上記の数値・出力維持は保存した試行candidateの証拠であり、現在の既定経路の変更ではない。D2D kernel自体を低下原因と断定せず、source／差分／ordinary／traceを保持する。

### OUT-2026-09-10-P835-R56: MLP residualと次段RMSNormの融合

- classification: finite scopeのN0候補。既存ResidualRmsNormのBF16 Add丸め・同じreduction／scale／epsilonと2出力を維持し、MLP後の63 inter-layer pairとfinal1 pairへ適用する。新native kernel／ABIなし。
- scope: R55と同じexact Qwen3.8 target。attention64とMLP64を別集計し、既存4B／MTP companion／adapter／multimodal条件は維持する。finalノードのpre-norm hiddenはfused output0、normalized hiddenはoutput1を読む。元tensor IDと各consumerを維持する。
- status: graph／core focused host検査と両GPU通常8192/128をPASS。128token／text hashとMTP採否はR55に一致、HIP-only／cleanup0。V3,648／R3,456の追加dispatch削減を確認したが、R9700単回decode24.435は目標未達で速度改善を認定しない。native数値証拠の不変対応は`r56-native-evidence-mapping.json`、新graph／runtime出力は`r56-gfx1030-output-comparison.json`／`r56-gfx1201-output-comparison.json`、詳細は[Phase83.5計画](../plans/archive/2026/09/1-10/phase83-5-llama-guided-performance.md)。
