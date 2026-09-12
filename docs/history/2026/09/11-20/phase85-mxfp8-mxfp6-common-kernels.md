# Phase85: MXFP8／MXFP6共通カーネル改善の履歴

## 2026-09-13: 計画の新設

ユーザー指示により、既存Phase85・86を後回しにし、新Phase85をMXFP8／MXFP6共通kernelの改善とした。
広範な行列形状で最適化した後、モデル本体量子化だけでなく、KVキャッシュとMTPでのMXFP利用を通した
推論効果を検証する。旧Phase85を86、旧Phase86を87へ繰り下げた。

専用active planを作成し、main-planの最新決定・進捗表と既存ロードマップの現在の順序・後続範囲を同期した。
過去の日付の決定・測定履歴は当時の番号を保持し、最新の再編を優先すると明記した。

計画は両RDNAのcodec／matmul共有部分と既存MXFP8 KV append／attentionを区別し、実在shape、合成矩形、
未整列・selector境界から三用途の通常CLI/API比較へ進む。MXFP6 KVの新設は含めない。
測定集合・回数は実装開始時に確定する提案として記録し、新しい速度倍率や独立承認段階は追加していない。

起点の公開HEADは`c2667ad885512329977d5883f3ea7b55dcfe2838`。着手前から存在する
AGENTS.md／THIRD_PARTY_NOTICES.md／provenance方針／main-planの取込み記録変更は維持した。
今回の計画作成では外部コードの取込み、kernel実装、GPU実行、常駐service変更、commit／pushは行っていない。
性能改善・数値成功・既定採用の結果はまだない。

## 2026-09-13: 実装開始・基準測定

追加指示「Phase85を完了する」により実装・検証・公開まで開始した。
形状manifestは`ci/matrix/phase85-mxfp-shapes-v1.json`の87件（MXFP8 44、MXFP6 43）へ具体化した。
小形状10件は全出力oracle、大形状77件は行・列・tail境界の明示sample oracleと全出力の非有限値／repeat digestを使う。
大形状のsample検査を全出力の数値oracle成功とは呼ばない。operator runnerは19 host test、
同形式・同一入力の品質capture runnerは4 host testに成功した。

起点nativeと新しい計測harnessを組み合わせた`baseline-gfx1030-r1`／`baseline-gfx1201-r1`は、
ソースのbuild前後一致とtarget別binary hashを保存した。証拠は`.local-artifacts/phase85/`に置く。
operatorの実行は3 warmup＋13 measuredである。準備時のjob manifestにある`measured_per_case=10`は
runner内warmupの見落としによる注記誤りで、実際の引数・reportは13反復を示す。
候補測定前に`progress.json`で訂正し、前後とも全13 measuredを同条件で集計する。

両GPUでMXFP8の先行43件、MXFP6の43件はoperator oracle・HIP dispatch・repeat・cleanupを成功した。
MXFP8の語彙幅248,320、K=5,120の行は、1.27GB weightの一括uploadがbackendの1GiB transfer上限へ達して
両GPUで失敗した。kernel実行前のharness不具合として保持し、転送を上限以下のrangeへ分割する修正後に当該行を再測定する。
R9700の測定終了・失敗時は既存serviceを元のunit/binary/configで復帰し、health／readyを確認している。

既存gfx1030 MXFP8 half2 ID55のforce比較では、通常ID22へ戻る6形状を独立に測定した。
N=17,408、K=5,120のM=128/129/1024/2048はoperator時間を約3.80/2.79/11.40/14.28倍改善した。
N=32,768、K=2,048、M=128は4.71倍、N=1,024、K=17,408、M=128は3.70倍だった。
独立oracleの最大relative errorは0.00344以下だが、旧row8と出力digestが異なる加算順変更であり、
現時点では通常selector採用・full-model品質・推論速度の成功には昇格していない。

共有codecのwave scale／packed group load、small-Mの共通MMQ、MX weight＋MXFP8 KVの通常入口をdraft実装中。
後者はcore graphが既に表現可能だがCLI/serverに未検証scopeの拒否が残っていたため、
直接graphによる数値・資源検査と公開入口の限定変更を合わせて検証する。

## 候補の比較と統合中の状態

- 起点87形状に加え、selector境界72形状、M1境界とMTP small-Mの54形状を別manifestへ固定した。
  `candidate-operators-gfx1030-r4`／`candidate-operators-gfx1201-r4`は87/87成功した。
  large-Mではgfx1030 MXFP8のN上限を32,768へ拡張し、gfx1201は整列済みdirect variantを維持しながら
  未整列Nをstaged WMMAへ接続する候補を実装した。V620の小N・短Kでは退行したため下限の既存条件を維持する。
- M1のwave/group共有はMXFP6で両GPUとも退行し撤去した。MXFP8はgfx1201の実形状で改善したため、
  K>=2,048／N>=1,024の候補に限定した。MXFP6は共有scalar unpackを3 byte読出しからbranchless 2 loadへ変更した。
  全64 code×4 slotの直接GPU byte検査を追加し、モデル本体・MTPでの最終収支を確認する。
- small-MのRows4／Columns8共通body（ID97/98）は両GPUでMTP重みのN/Kを用いたM2..4演算子ケースの数値・repeat digestを維持した。
  MXFP8専用のsmall-M load policyを通常cache loadへ変更し、既存large-Mのload policyは維持した。
  gfx1030のMXFP8 K=6,144／N=5,120はこの調整後も遅いため、退行する形状分類を旧providerへ残して統合する。
- 通常KVのwave共有候補は21ケースのappend／attentionと既存prefill oracleで出力一致を確認したが、
  短contextで両GPUとも遅く、causal attentionへの変更は撤去した。長contextの既存staged経路は維持する。
  KV比較probeの追加時には、decode splitの2 dispatchを旧prefillの1 dispatch検査が拒否し、cleanupへ波及した。
  pathごとの厳密metadata検査と全completionのdrain/releaseへ修正し、両targetで再実行を完了した。
- 最初のgfx1030 wave候補のMXFP6 M=1,024／K=5,120／N=17,408では、repeat4のdigest不一致を一度記録した。
  失敗は`.local-artifacts/phase85/candidate-operators-gfx1030-r1/`に保持している。改善前・候補の追加32反復は
  いずれも同じdigestで成功し、ID57と先行quantizerの命令列・barrier・resourceも一致した。
  この一件をkernel変更に帰属せず、runtime／一時的要因は未特定とする。r4の同形状を含む87件は成功したが、
  失敗を成功へ読み替えない。再発時に差分要素と同じGPU出力の再読出しを記録する診断を追加した。
- 改善前のbody 10条件、MTP off／BF16／MXFP8／MXFP6の4条件、body＋KV quality 4構成を両GPUで取得した。
  CLI/serverのMX weight＋MXFP8 E4 KV限定guardと、量子化companionを受け付ける状態診断を実装した。
  **最終実装の通常推論・MTP状態・言語/API比較と、公開／CIはまだ完了していない。**
- integration reviewは一回実施した。推論集計のdraft側HIP/fallback/dispatchとbody session cleanupの確認漏れを修正し、
  fault injectionとfocused re-reviewで解消を確認した。新しい全面reviewや完了条件は追加していない。

再開時は`.local-artifacts/phase85/progress.json`、各runの`execution.json`とbuild `identity.json`を確認する。
既存R9700常駐serviceは元のunit／binary／設定で復帰済みで、`readyz` 200を確認した。

## 最終候補r6の公開入口検証

- `candidate-gfx1030-r6`／`candidate-gfx1201-r6`はbuild成功・build前後source一致を確認した。
  通常APIのMXFP8／MXFP6本体＋FP16 KV、BF16本体＋MXFP8 E4 KV、MXFP8／MXFP6本体＋MXFP8 E4 KVの
  計5構成を各GPUで実行し、各5要求（non-stream、SSE、cancel、recovery、別要求）を成功した。
  全要求のHIP dispatch、fallbackなし、要求状態／workspaceとsessionのcleanupを確認した。
- MXFP8／MXFP6 MTP companionの通常APIも両GPUで成功し、短文・SSE・cancel後recoveryに加えて
  8,192入力／128出力を完走した。証拠は`.local-artifacts/phase83/phase85-final-api-*-mtp-*-smoke-r6/`。
  言語・タスク別採用率と最終性能比較は別runで集計する。API実行中のwall値を性能代表値へ使わない。
- 現行sourceから再buildしたnative public runtime host test、operator harness 19件、quality 4件、
  CLI/server guard、Phase84.5診断の2 host testを成功した。古いcache binaryだけを実行した先行確認は
  staleと区別し、現行source検証の代わりにしない。
- `gfx942:sramecc+:xnack-`／wave64／Code Object V6ではmatmul、KV state、causal attentionの3翻訳単位を
  compile-onlyで確認した。実機・数値・性能の成功へ拡張しない。
- 最終静的検査でRust/C++の書式とClippyの指摘を修正した。C++変更は空白以外の一致を記録した。
  累積確認ではMXFP6 small-M既定が旧MMQ／row8／tiled16のforce指定に先行する問題を見つけ、
  旧forceの優先順位を復元した。通常実行とPhase85 force単独の選択は維持し、関連host検査を追加する。
  最終性能用buildはこの修正を含む候補へ更新する。

## r7のhost検査と推論比較の途中結果

force優先順位修正を含む`candidate-gfx1030-r7`／`candidate-gfx1201-r7`をbuildし、sourceの前後一致を確認した。
H1は1,570件、H2は38件を成功した。H0は書式・Clippy修正後626/627件が成功し、残るcanonical public-runtime
source hashの未同期を修正して`validate_matrix.py`を成功した。全H0再実行は追加せず、失敗箇所の解消を記録する。

通常APIは両GPUで全20比較が完了した。言語／タスクの本測定は36条件×3 companion形式×2 GPUの216条件である。
採用token合算の率は次のとおりで、生成履歴の異なる形式間の率差をkernel誤差や速度改善へ直結させない。

| target | BF16 MTP | MXFP8 MTP | MXFP6 MTP |
| --- | --- | --- | --- |
| gfx1030 | 2,617/3,930（66.59%） | 2,619/3,935（66.56%） | 2,609/3,953（66.00%） |
| gfx1201 | 2,628/3,919（67.06%） | 2,639/3,899（67.68%） | 2,597/3,972（65.38%） |

r6 APIとr7の差は書式、operator計測器の等価な条件式、MXFP6の比較用force優先順位、host検査に限定される。
APIでは該当forceを指定せず、通常選択とGPU演算は同じであることをsource mappingへ記録した。
r6 APIをr7 binaryの実行と偽らず、この確認範囲で再利用する。最終性能比較はr7 binaryで実行する。

R9700の本体・KV速度比較12条件は完了し、全条件で生成token列とVRAM使用量が一致した。
MXFP8／MXFP6本体＋FP16 KVのdecodeはそれぞれ約3.8〜4.6%／4.7〜5.2%改善し、
MXFP8 E4 KVとの併用でも約4.4%／5.1%改善した。prefillとBF16 KV対照は概ね横ばいだった。
V620も本体・KVの12条件を完了し、全生成token列とVRAM使用量は一致した。速度はおおむね±1%で、
明確な本体推論の改善は確認していない（decodeの範囲は約-1.0%〜+1.5%）。
両GPU×本体MXFP8／MXFP6×FP16／MXFP8 E4 KVの8品質構成では、各10ケース／20 logits行、
合計160行が同形式の変更前と一致した。最大absolute差とKLDは0、top-1一致率は1.0である。
これは同形式の最適化前後比較であり、BF16比の量子化品質が改善したとの主張ではない。
量子化MTPの状態照合・全体速度、最終operator分布、公開CIの確認は引き続き別に記録する。
集計器では、本体の形式と後続のKV形式を区別し、BF16／MXFP6本体＋MXFP8 KVをMXFP8本体へ誤分類しないよう修正した。

## 最終採用と結論

最終sourceはr10で、[集約結果](../../../../../ci/matrix/phase85-mxfp-common-kernel-results-v1.json)へ
入力・build identity、中央値／MAD、shape別比較、推論、品質、API、状態の範囲を記録する。
MXFP8の実行kernelとquantizerはr7/r10で機械語・サイズ・resource descriptorが一致し、r7測定をcomponent単位で再利用した。
MXFP6は下記scope修正後に影響する演算子、通常推論、品質、状態を再測定した。r7の値をr10実行と偽らない。

| 採用 | 最終scope |
| --- | --- |
| A: Rows4／Columns8共通MMQ、ID97/98 | M2..4、K>=2048かつK%32=0、N1024..32768。MXFP8 gfx1030では整数K/N=1を旧providerへ残す |
| B: 既存MXFP8 large-M provider | gfx1030 half2のN上限32768、既存下限を維持。gfx1201はN tailをstaged WMMAへ、aligned directを維持 |
| C8: wave block scale／value load | gfx1201 MXFP8 M1、K>=2048、N>=1024。加算順を維持 |
| C6: packed E3M2 read policy | 2-loadはMMQとgfx1201 scalar reductionへ。scalarの他targetとtiled16にはlegacy24／3-byteを維持 |

C6を全consumerへ一律適用したr7は、gfx1201 M8/K1024/N1025を約2.07倍遅くした。
32反復のcontrol/candidate/controlで再現したため、最終r10ではtiled16を旧readerへ戻した。
同じ形状はcontrol `133,684.5 / 133,684 ns`、r10 `133,583.5 ns`となり退行を解消した。
gfx1030 M1の3実形状も旧readerへ戻し、基準を挟む32反復で同等の時間へ復帰した。
最初の広範測定だけで観測した2.36倍の改善は、この反復対照では再現せず、採用根拠にしない。

最終operator集計には、gfx1201 MXFP6 M129/K5120/N17408の大きい時間変動も残す。
r10は13反復で約1.36〜18.81 ms、中央値6.493 msとなり、基準1.368 msに対して4.75倍遅い。
同じID48のr7中央値は1.361 msであり、r7/r10の機械語・資源情報と出力digestは一致する。
この実測を削除せず最悪退行と分布へ含めるが、変更したreaderの効果やkernel変更に原因を帰属しない。
全shapeの安定した高速化を示す結果ではなく、上記の推論効果は別runの同条件比較に基づく。

通常MTPのprofileでは、MX companionはM1 decodeとlarge-M prefixを使い、Rows4はdispatchしなかった。
このためAをMTP高速化へ帰属せず、通常CLIの515入力／17出力、明示chunk512による末尾M3で確認した。
両GPU・両形式でID97/98を実dispatchし、前後token列とVRAMを維持した。
この条件のprefillはMXFP8/MXFP6がV620で約10.0%／3.5%、R9700で約24.3%／11.3%改善した。
通常の512/128・2048/128等ではR9700 decodeがMXFP8で約3.8〜4.6%、MXFP6で約4.7〜5.6%改善した。
MXFP8 E4 KV併用でも約4.4%／5.5%改善した。V620の通常長の本体decodeは概ね横ばいである。

量子化MTPの最終8192/128は、V620 MXFP8/MXFP6 decodeが`20.899 / 22.610 tok/s`、
R9700が`32.466 / 29.822 tok/s`。同じ候補のBF16対照は`25.332 / 34.775 tok/s`で、
量子化MTPはBF16を上回らず、BF16既定を維持する。形式内の前後差とBF16比較を混同しない。
KV向けwave候補は短contextで退行したため不採用とし、既存append／attentionを維持した。
モデル本体＋KVの改善はKV kernel自体の改善とは呼ばない。

量子化MTP状態の主証拠はV620 normalとR9700 attention-only対照である。R9700ではnormalと対照の
prompt、prefill hidden、draft token／logitが一致し、attentionの演算順だけを揃えると採用0/1/2後の
KV、hidden、次計算、companionの差が消えた。normal FAILとその原因を保持し、main attentionは変更しない。
先行したforce-baseline併用の診断はMXFP8 codingのdraft列も変えるため、同一履歴の主対照にしない。

最終確認中のharness不備も残した。旧row8 forceをRows4として検査していた期待値を修正し、
実ID／symbol／gridを照合する21 host testと両GPUのforce probeを成功した。
CLI profileの0+1は既存protocolが拒否したため、1+3へ訂正して再実行した。profileの時間は代表性能値へ使わない。
これはkernel失敗の修正ではなく、検証器と実行条件の修正である。

CPU検証はH0の書式／Clippy／manifest等、H1 1,570件、H2 38件と関連native host testを確認した。
両targetのHIP build、codec、新旧reader、演算子と推論を実機確認し、gfx942は3翻訳単位のcompile-onlyに限定した。
既存R9700 serviceは元のunit／binary／設定へ復帰し、health／readyを確認した。raw profile、binary、modelはGitへ追加しない。
公開前のJSON検査で、public-runtime manifestを参照するRMSNorm CI契約のhashも同期し、検査を再実行した。
公開CIの成否は本変更のGitHub Checksで確認し、ローカル測定の成功へ混ぜない。

残差はPhase86へ引き継ぐ。MTPのtarget検証・sampling等の費用は今回のMX kernel改善では変えず、
全shape／全modelの一律高速化、BF16比品質同等性、MI300X実機、MXFP6 KVは今回の結果として主張しない。

[計画](../../../../plans/archive/2026/09/11-20/phase85-mxfp8-mxfp6-common-kernels.md) /
[メイン計画](../../../../plans/main-plan.md)
