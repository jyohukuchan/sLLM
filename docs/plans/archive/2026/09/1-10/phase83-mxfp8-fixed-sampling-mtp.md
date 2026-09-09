# Phase 83: MXFP8 KV・固定sampling・MTPの実用統合

> 状態: 実装・検証・比較記録完了（2026-09-09）。同日のユーザー指示で正しい実装の完成までをPhase 83とし、追加最適化・速度目標をPhase 83.5へ移した。

## 受入条件と範囲

2026-09-08のユーザー決定を実行する。正本は[main-plan](../../../../main-plan.md)の
「Phase 83の実装完了とPhase 83.5の速度目標」と[ロードマップ](../../../../active/2026/09/1-10/phase76-qwen38-27b-nvfp4-priority-roadmap.md)のPhase 83節。
以下はその実行記録であり、追加の完了条件を設けない。

- Qwen3.8 27B NVFP4、standard OCP MXFP8 E4 KV、固定GPU sampling（T=1、P=0.95、K=20）とBF16重みMTPをCLI/APIで併用する。
- 8,192入力／128確定出力、single GPU・batch=1・MTP有効時の正しい長文生成を両GPUで確認する。
  V620 prefill 200／decode 20 tok/s、R9700 500／25 tok/sの達成はPhase 83.5へ移し、Phase 83は実測と未達差の記録までとする。
- Phase 82公開HEAD `63ef9057f6265d99e38b254b8fb31d0b426859a4`を同条件の基準とし、KV差とMTP効果を分離する。
- 固定samplingの正しい実装、accept/reject・replay・RNGとKV/GDN/MTP状態、SSE・cancel/recovery・unloadを確認する。
- 2026-09-09ユーザー決定: MTPなしとの出力完全一致は必須としない。同一BF16参照に対する精度劣化が同程度ならMTPによる出力差を許容する。
  この方針はPhase 83.5にも引き継ぐ。BF16参照・入力・評価指標・許容差を記録し、samplingのばらつきと数値差を区別する。
  token不一致だけで失敗にせず、単一生成例やkernelのBF16出力一致だけでモデル品質の同等性を認定しない。
  要求履歴に依存する文章崩壊や状態・sampling実装の不具合は許容対象外とする。
- 32 GB級VRAMへ収容し、GTT spill／CPU fallback／常駐FP16 mirrorを作らない。
- tools完全対応、vision、全モデル、TP、batchingは対象外。Phase完了時のcommit・push・公開CI確認を行う。

## 分割後の作業方針（2026-09-09ユーザー決定）

- MTP有効時の要求再利用で文章が崩れる不具合を解消し、CLI/API・状態管理・数値の正しさを完成させる。
- 新規の速度探索はPhase 83.5で行う。既存の追加最適化は正しさと採用条件が確認できたものを残し、未検証候補は切戻しまたは無効化して引き継ぐ。
- 以下の探索・不採用記録は履歴として保持し、未完了の速度探索をPhase 83の完了条件にしない。
- 両Phaseでそれぞれ完了時のcommit・push・公開CI確認を行う。後続順序は83→83.5→84→85。

## 実行順と進捗

1. **完了: 比較入口と初回比較。** Phase 82の旧benchmarkはFP16固定、合成replay、EOS停止なしだった。
   既存モードは履歴再現用に維持し、MXFP8選択、実8192-token入力、通常EOSを扱う測定を追加する。
   同じ計測入口をPhase 82 core/nativeと候補版へ接続し、harness変更とruntime変更を区別する。
2. **完了: 共通MXFP8経路。** append／attention／context growth／Graphの形式別処理を統合した。
   gfx1030／gfx1201の実GPUで、Qwen38の24 query head／4 KV head／head幅256、長さ31→32→33の
   MXFP8 append-chainを独立量子化・softmax oracleと比較した。両targetでBF16出力の最大ULP差0、
   repeat、cancel、cleanupも成功した。これはnative連続実行の証拠であり、モデル全体やMTP統合の完了証拠ではない。
3. **完了: MTP／公開入口。** 既存MTP resident／graphと固定GPU samplerを接続し、gfx1030専用APIの制限を解消した。
   APIのMTP選択が`requires_logits`で固定samplingまで除外していた条件を修正し、GPU selectorで処理可能な
   samplingをMTPへ接続する。固定profile／seedなし／host logprobs／greedyのfocused host回帰検査が成功した。
   R9700の既定CLI/API長文と対話は成功した。V620もID91 public launcher修正後のr26長文と対話が成功した。
4. **追加探索はPhase 83.5へ移管: 共通prefill性能。** 保留中の単純DP4A逐次和を性能だけで既定化せず、誤差を改善するreduction候補を評価する。
   補償加算候補ID87の実kernelをgfx1030／gfx1201で実行し、非整列shapeを含む16ケースで
   独立数値oracleとの最大BF16 ULP差1以内、非有限値なし、repeat一致を確認した。小型shapeは全要素、
   大型shapeは32点のoracle比較であり、モデル全体の品質・速度・既定採用は未確認。
5. **完了: 統合検証。** target-only／MTP、両GPUの代表性能とAPI lifecycleを確認し、検証binaryと最終sourceの対応を記録する。
6. **公開手順。** 差分に応じた検証と履歴・証拠整理を完了。commit・push後の公開CI結果は当該commitのChecksと完了報告へ記録する。

## 2026-09-09の測定状況

- Phase 82 runtimeへ測定用harnessだけを移植した独立worktreeで両targetのbinaryを作成した。
  coding入力は既存Rust chat template／tokenizerで8,192 tokenとし、token列SHA256は
  `855240c09609a19b9c1124043b763ecc97e3cfde84a16acfaba445a0f8c83b23`。
- 両GPUのMXFP8・MTPなし、1 warmup＋3 measuredの基準測定が完了した。
  V620-Aの中央値はprefill **8.223322 tok/s**、decode **8.061763 tok/s**、
  R9700はprefill **10.707189 tok/s**、decode **11.012592 tok/s**。
  全measured行が128公開token、HIP-only、fallbackなし、解放後zeroで完了した。
  同一fixtureを使用したPhase 82通常設定の基準であり、目標達成値ではない。
- V620-Bの探索測定（MXFP8、MTPなし、ID87 opt-in、warmupなし／単回）は8,192／128で
  prefill **92.8407 tok/s**、decode **7.9499 tok/s**、128公開token、HIP-only、非有限logitなし、cleanup成功。
  目標未達であり、異なるGPU個体で進行中のPhase 82基準測定との正式比較や既定採用の証拠にはしない。
  候補binary SHA256は`f050076dde5d8b39367674e81ac0ceb2a33e2e039790f798a2124e4ef576f78f`。
  同じ探索条件のkernel traceを取得し、prefill／decodeの残る費用を切り分ける。
- sampled MTPはwidth1〜3の一括target verification、採用prefixのqueue、補正／bonus、状態rollbackを実装した。
  prefixはhead実行を省略する最大1024行のbatch、embedding／FP8 headはtarget residentとの共有へ変更した。
  host検証済みだが、最初のGPU試験ではuntied FP8 headをtied用node名で構築する不整合によりロードが失敗した。
  エラー後の解放は成功。node生成／binding検査の修正は実artifact host検査に成功した。
  GPU再試験は`mtp.embedding_norm.row.0`のtoken extent検査で失敗し、解放は成功した。
  行単位のMTP中間tensor aliasをowned tensorのextent検査から分離し、既存workspace bounds検査へ委ねる修正を行った。
  focused host検査は成功。修正版のGPU試験はresidentロードを通過したが、
  MTP prefill hidden hookがQwen3.5の幅2560を固定しており、Qwen3.8の幅5120を拒否した。
  解放は成功。周辺のhidden hook／buffer／graphの固定寸法を修正し、host検査が成功した。
  次のGPU試験は1024行graphから129行runtimeへのalias view変換で失敗したため、
  部分入力時のrow alias bindingと未使用行の実行範囲を修正した。
  129／1024のhost検査とr7 GPU短行が成功した。129入力は5確定token（EOSを含み、公開4 token）で終了し、
  MTP width2で2／4 draftを採用した。既存target-only短行と生成token hashが一致し、解放もzero。
  短行は8192／128の受入証拠ではなく、長文MTPの測定を開始した。
  MTPの長文採用率・速度・公開入口の成立は未確認。
- 実測用にR9700常駐serviceを接続要求がない状態で一時停止した。GPU検証後の復帰が残る。

## 探索メモ

- kernel traceでV620 prefillのgeneric MXFP8 attentionが約60.188秒、decodeのwave-split attentionが
  約8.201秒（decode GPU時間の約54%）を占めることを確認した。model loadをprefill区間から除いて解析した。
- ID87＋既存GQA6 QTILE4のV620-B探索行は **211.0486 prefill／7.8054 decode tok/s**。
  8,192／128、MTPなし、明示opt-in、warmupなし／単回であり、MTP有効・既定設定の完了証拠ではない。
  QTILE4の数値分類は[N1根拠](../../../../../compatibility/numerical-output-changes.md)へ記録した。
  V620とR9700でquery127／128／129・prefix0／31／256の独立MXFP8 oracle検査が成功した。
  両targetとも最大BF16 ULP差1以内であり、当該fixtureでは既存providerとQTILE4もbitwise一致した。
- 小M matmul公開API probeは、現行NVFP4 row8経路のM2〜4がM1逐次より遅いことを示した。
  M1の既存reductionを行ごとに別gridへ展開するID88をopt-inで実装中。scratchのM2／3／4、
  K5120/N17408とK17408/N5120、各2 seedの12ケースは数値・finite・repeat検査に成功した。
  V620の公開runtimeでもM1〜4、NVFP4 wide／downとFP8 GDN／headのprobeが成功した。
  wideのID88はM2 0.330887 ms、M3 0.464328 ms、M4 0.604291 msで、M1逐次比0.847／0.793／0.774。
  これは演算単位の証拠であり、MTP全体の実測は未完了。

- 補償加算ID87の読込tile Kを32から128へ増やすローカル候補をV620-Bで確認した。
  16ケースの数値・repeat検査は成功。M1024のwideは約10.86〜10.93 ms、downは約10.99〜11.01 msで、
  K32候補の約11.04／11.57 msからの改善は小さい。モデル全体の目標との差を単独で埋める案ではなく、
  productionへの反映は保留し、kernel traceで大きい費用を優先する。

- attentionの同じ8区間・加算順をstage1／stage2へ分けたM1候補は、V620でlength1023／1024／1025／
  8191／8192／8193の全行が旧経路とbitwise一致した。8192のGPU時間は3.29677→3.18341 msで改善は小さい。
  M2〜4対応とE8 scaleのwave内共有を追加した。8192ではM1〜4の全行が旧M1逐次とbitwise一致し、
  独立oracleとの最大BF16 ULP差1以内。M3は9.90302→3.18574 ms、M4は13.2986→3.21454 ms。
  公開runtimeでもV620-Aのlength1023／1024／1025 × M1〜4を検査し、1023は従来ID3、
  1024以上はID80・2 dispatch・workspace198144×M bytesの選択を確認した。oracle最大ULP1、repeatと解放も成功。
  モデル全体のr6試験ではRust側のmetadata validationがID80を拒否し、decode1で失敗した。
  エラー後の解放もresource busyとなったため、Rust受入条件と失敗時のcompletion待機を修正した。
  ID80の正しいmetadata受入・不正条件拒否のhost検査とr7のモデル全体検査が成功した。
  V620-Aの8192／128、MTPなし、単回探索は **214.948288 prefill／8.604439 decode tok/s**、
  128公開token・cleanup zero。MTP有効時の目標達成は未確認。

- R9700向けID89 WMMA補償加算候補を実GPUで10ケース検査し、独立long-double oracleとの
  最大BF16 ULP差0、再実行bitwise一致を確認した。M1024 wideは8.499〜8.629 ms、downは9.510〜9.768 ms。
  旧ID64との平均時間比は1.050倍だが、ID64は数値判断が保留された候補であり正式baselineではない。
  既定採用はPhase 82通常ID59との比較とモデル全体の実測で判断する。現時点ではopt-inを維持する。

- R9700 r6の8192／128、MXFP8・MTPなし、ID89＋QTILE4 opt-in、warmupなし／単回は
  **364.004804 prefill／9.760341 decode tok/s**、128公開token、cleanup zeroで完了した。
  prefillの500 tok/s目標には未達。ID89 includeのFP contraction pragmaが後続関数へ作用していたため、
  candidate関数内だけへscopeを修正した。r6の数値は修正前の探索identityとして残し、最終採用証拠にはしない。
  このbinaryのkernel traceを取得し、prefillの残る費用を調べる。profileは診断専用である。
- 新しいnative includeのCMake／Cargo依存登録、G2／H3 source inventory、path-to-suiteを更新した。
  H3 contracts 31件、runner43件、G2 schema6件、matrix検査とhip-sys host checkが成功した。
  source変更後の最終hash同期と公開CI確認は完了時に行う。

- ID89のstage Kを32→64／128へ増やすscratch候補をR9700で比較した。
  scale読込をblockDim strideへ直した最終候補は全10ケースでoracle最大ULP0／repeat一致。
  stage64はM1024で約3.8%改善する一方、M63〜65で約20〜22%遅く、stage128は約25〜67%遅かった。
  初回の大きいstageはscaleの256要素だけを読む処理が不足してrepeat不一致となり、修正版と区別して記録した。
  現在の500 tok/sとの差を解消する候補ではなく、productionはstage32を維持する。

- V620 r7のMTP有効8192／128がwidth1／2／3で完了し、全て128公開token・解放zeroだった。
  単回探索はwidth2（V620-A） **178.694696 prefill／7.501187 decode tok/s**、採用71／112 draft、
  width3（V620-B） **173.878585／7.062648 tok/s**、採用79／147 draft。
  width1（V620-B）はprefill47.759597秒／decode18.537931秒で、幅を狭めるだけでは改善しなかった。
  width間のGPU個体差があるため最終幅選択の反復比較ではない。MTP有効時の目標は未達。
- V620 width2 traceをtarget prefill／MTP prefix priming／decodeへ分けた。
  primingはBF16 tiled16行列積が **6.6567秒／64 calls**、attentionが約0.7172秒だった。
  decodeはID80 stage1が約5.1735秒、FP8 half2の32×32が約3.2870秒、32×64が約2.5423秒。
  MTP priming終端は大きいembedding batchから次の単一行draftへ戻るGPU dispatch境界による推定であり、
  profileを正式throughputへ代用しない。BF16 prefillとFP8 small-Mの演算共通候補を優先する。
- R9700のMXFP8 QTILE8 scratchはQTILE4とbitwise一致し、独立oracle最大ULP1以内だったが、
  query127〜129と1024・prefix0／4096／7168で約1.12〜1.65倍遅かった。QTILE4を維持する。
- APIの厳密な`prompt+max_new_tokens`容量でMTPの先読みが末尾を超える問題を修正した。
  target／MTP残容量からdraft幅を制限し、target残1行は逐次samplingへ切り替える。境界host検査が成功した。
  正規CLIのQwen3.8 artifact入口は未実装だったため、既存server backendを再利用する入口を実装中。
- V620-BのBF16 prefill scratchでは64×64／K32 tileの共有weightを転置し、
  M1024のFC／MLP up／downでtiled16比60.7%／68.0%／67.2%短縮した。
  全10ケースで全出力のcontrolとの差0 ULP、各ケース32要素の独立oracleとの差0 ULP、repeat一致。
  M63／65・K37・N37では約2.2倍遅いため、大型形状に限定する公開経路への統合を進める。
  これは演算単体の探索結果であり、モデル全体の速度達成を意味しない。
- R9700 API smoke r1はharnessが未対応の`max_tokens`を送ったためHTTP400となり、推論要求は実行されなかった。
  公開仕様の`max_completion_tokens`へ修正してr2を実行する。r1は失敗として保存し、解放zeroだけを確認した。
- API smoke r2は通常text要求で17 token生成に成功し、MTP draft12／採用10、全dispatch HIP、fallbackなし、解放zeroを確認した。
  続くSSE要求はharnessの未対応`stream_options`によりHTTP400。公開仕様を確認してこのfieldを除き、
  8192／128の出力数はshutdown auditの対応要求で確認する形へ改め、r3を実行する。
- API smoke r3は通常text／SSE／途中cancel／次要求の回復を実GPUで実行し、全dispatch HIP、解放zeroを確認した。
  8192入力の要求は自然EOSにより11出力で終了したため、128出力の受入条件は未達としてrun全体をFAILで保存した。
  EOS停止を無効化せず、benchmarkとの入力・sampling条件を照合する。
- MTP blockの部分確定後、再計算したtarget hiddenを次のdraftへ渡すようfrontendを修正した。
  元blockのcached hiddenと異なるreplay値を使うhost検査を追加した。最終binaryでのGPU確認は未実施。
- CLIのQwen3.8 generate／chat入口を追加し、既存production backendへ接続した。
  CLI 89件、server lib131件（既存ignored1件）のhost検査が成功した。
  generateのMTP／fallback表示は要求監査から取得し、明示shutdownの失敗をCLI終了結果へ伝える。
  実モデルでのCLI確認は未完了。
- R9700 ID89の2つのK16 termをstage内で合算してからKahan更新するscratchは、10ケースで独立oracle最大ULP0、repeat一致。
  ID89比M1024 wideは1.4%、downは4.1%短縮に留まった。productionには未採用であり、attention／FP8行列積を優先する。
- API r3とbenchmarkは入力message／render方針／seed123／固定samplingが一致する一方、
  APIはMTP2・state容量8320・prefill2048×4、R9700 r6 benchmarkはMTPなし・容量9563・prefill1024×8だった。
  APIの短い反復出力を正しさ確認済みとは扱わず、同じbinaryでMTP有無を切り替えて原因を切り分ける。
- R9700 API r4はr8 binary／同じ入力・seed・context8320・prefill2048でMTPだけを無効にし、
  128 tokenの一貫したコードレビュー文、HIPのみ、解放zeroで成功した（要求wall35.644秒）。
  r3の反復文と早期EOSはMTP経路に絞れた。replay hidden修正版で再確認し、未解消なら一括target検証を調査する。
- 追加切分けr5では同じr8 binaryのMTP有効・起動直後の長文要求が128 tokenの正常文で成功した。
  draft104／採用76、要求wall42.626秒、HIPのみ、解放zero。r3との違いは先行する短文／SSE／cancel／回復要求の有無であり、
  MTP演算自体だけでなく要求間の状態・資源再利用を調査する。r9（nativeはr8維持、replay hidden修正＋CLI追加）で同じ要求順を再検証する。
- r9 API r6は通信／128出力／HIP／解放のharness検査を通ったが、文章は断片の反復に崩れ、draft253／採用0だった。
  実用文章生成としては失敗とし、別の`text-assessment.json`に元execution hashと判定を保存した。
  API状態管理の作業単位を再計画する。進行中のr9起動直後長文との切分け後は、原因を特定・修正するまで
  同じfull-model検査の反復や性能matrix拡大を行わない。kernel単体の独立作業は継続する。
- r9起動直後の長文要求r7は128 tokenの正常文で成功し、draft104／採用76、wall42.747秒、解放zeroだった。
  r8／r9とも起動直後は正常で、短文／SSE／cancel／回復後に崩れるため、要求間の資源再利用・初期化を原因調査の対象とする。
- R9700の実CLI短文要求は26入力／17出力、MTP draft12／採用10、全dispatch HIP、fallbackなし、明示shutdown成功を確認した。
  local harnessはCLI reportの`result`階層を見落として失敗表示となったが、保存済みstdoutと終了コードを再検査して
  `assessment.json`へPASSと元ファイルhashを記録した。GPUの不要な再実行は行わない。
- ID89のLDS leading dimensionを32→48へpaddingするscratchをR9700で試した。
  10ケースでbaselineとの出力hashが一致し、oracle差0 ULPだったが、M1024 wide／downは0.73%／0.99%遅く、
  LDSは7,680→10,752 bytesへ増えた。性能改善がなく不採用とし、productionは変更しない。
- V620-Aの実CLIでも26入力／17出力、MTP draft12／採用10、HIPのみ、終了コード0を確認した。
  両targetの短文CLIは確認済みだが、長文・対話・要求再利用を含む実用完了とは区別する。
- MTP要求がresident queueを共有する構造を見つけ、deferred completionの選択とは独立して要求専用queueを作る修正を追加した。
  独立queue、cancel後の新要求、重みの再uploadなしを確認するhost test1件が成功した。r10で同じAPI要求順を検証する。
- r10要求専用queue候補でも短文／SSE／cancel／回復後の長文は9出力でEOSとなり、問題は解消しなかった。
  このqueue変更と専用testは戻し、r10 build／API結果を不採用試行として保存する。共有queueを原因と断定しない。
  次の切分けはnative buffer／GDN stateの初期化と再利用に限定し、具体的な原因候補が得られるまでfull-model反復を止める。
- 静的調査ではfresh linear stateのzero初期化、completion参照を保持した解放、request-local prepared cacheを確認したが、直接原因は未特定。
  比較を「起動直後／要求履歴あり × MTPなし／あり」の4ケースへ整理すると、要求履歴あり・MTPなしだけが未確認だった。
  同条件の不要な再実行とは区別し、この未確認1ケースをr9 binaryで診断する。現時点ではMTP固有の不具合とは断定しない。
- ID91〜93後のH3 native/header hashと101-file digestを同期し、matrix／G2検査およびfocused pytest98件・666 subtestが成功した。
  H3 runner実compileはdirty checkout拒否で未実施。既存の両target直接compile証拠と区別し、最終候補で必要なCI検証を行う。

## 再開時の確認（2026-09-09）

- 保存済み `api-r9700-targetonly-sequence-r9/execution.json` を再確認した。
  r9 binaryで短文／SSE／cancel／回復の後にMTPなしの8,192入力／128出力は正常文、HTTP 200、HIPのみ、解放zeroで成功している。
  起動直後／要求履歴あり × MTPなし／ありの4ケースが揃い、文章崩壊は要求履歴あり・MTP有効のケースで観測された。
  前節の「未確認1ケース」は解消した。これは当該実装・条件での切分けであり、モデル品質全体の証明ではない。
- 要求専用queueでは解消しなかったため、native FP8 hipBLASLtの同一contextでのshape変更・再利用をモデルなしで診断し、
  並行してcoreのresident共有buffer／scaleとMTP初期化を調査する。いずれも原因仮説であり、まだ原因を特定していない。
- 中断前のBF16 scaled operand WMMA scratchは数値比較10ケースでID89と全出力一致したが、
  R9700でID89より約1.38〜1.88倍遅く不採用。productionへの変更はなく、結果は
  `.local-artifacts/phase83/bf16-scaled-wmma-gfx1201-r1/report-final.md` に保持する。追加の速度探索はPhase 83.5へ移管した。

- R9700のモデルなしFP8 public ABI診断（`fp8-same-context-r1`）は、headのK=5120／N=248320で
  fresh M2048、同一contextのM2048→26→17→2048、fresh M2048を比較した。全6行でsampled oracle差0 ULP、
  M2048の全出力hashは一致した。出力bufferをNaNで埋める追加条件でも同じ結果となり、解放zeroを確認した。
  このshape遷移ではhipBLASLtの再利用による破損は再現していない。内部activation／scale hashは未取得であり、一般的な無欠陥証明ではない。
- coreのresident共有／dynamic alias調査でも直接原因は特定できなかった。r9と同一runtimeにCPUへ既に取得済みのhidden／選択tokenの
  診断出力だけを加えたr11をscratchで作成し、fresh長文と要求履歴後の長文の最初の差を調べる。
  これは原因箇所を特定する診断であり、速度測定や最終受入証拠ではない。productionへの診断hook追加は行わない。

- BF16参照の在庫調査ではQwen3.8-27B BF16 safetensors（18 shard、重み約55.56 GB）とBF16 GGUFは存在するが、
  sLLMの当該BF16公開入口／model lockは未整備で、単一32 GB GPUには収まらない。
  現在のkernel BF16出力oracleおよびNVFP4モデル測定を、BF16 full-modelに対する品質同等の証拠とは扱わない。
  BF16とNVFP4でtokenizer／templateのファイルidentityも異なるため、参照比較ではtoken対応を確認する必要がある。
  新規BF16 bringup／TP／offloadをPhase 83の追加完了条件にはしない。現時点でBF16 full-model品質同等性は未証明である。

- r11のfresh／要求履歴後の比較で最初の分岐を特定した。8,192入力prefillの全hidden hashと最初の選択tokenは一致し、
  最初のverifyも入力 `[2,9988,314]`、全hidden hash、選択 `[5927,321,1510]` が一致した。
  最初のdraftは棄却され1行をcommitする。そのrollback／replay後、次のproposalとtargetの先頭行hiddenが分岐する。
  freshは正常な128 token／draft104・採用76、要求履歴後は128 tokenでも文章崩壊を再現した。
  通信検査と別に`text-assessment.json`へFAILを保存した。調査対象を最初の棄却後の状態復元・M=1 replayへ絞る。

- r12はtarget replayとMTP draftの入出力hiddenをscratchで追加記録した。fresh／要求履歴後でprefill、最初の2 draftの
  入出力hidden、最初のverifyは全て一致し、最初のtarget replayで初めて差が出た（hash `4a6c1029f37c3095` 対 `a411dff740461e11`）。
  その差が次のMTP入力へ渡される。後続draftの相違がtarget先頭行へ与える影響を推測する必要なく、targetの状態復元・M=1 replayへ絞れた。
  freshは正常文、要求履歴後は文章崩壊を再現した。native rewindの構造調査とreplayの層別診断を進め、原因未特定のまま修正済みとは扱わない。

- r13の層別diagnosticはlayer 0のhash取得後にCompletionPendingで失敗した。追加したmid-graph flushが
  deferred segmentをprofiledへ戻した後、そのsegmentを継続利用した診断コード側の問題だった。
  r14では診断readback後に元のcompletion modeでsegmentを再作成し、layer 0の各演算と全層末尾を
  最初のtarget M=1 replayだけで記録する。freshは正常128 token／draft104・採用76／解放zeroで完了し、
  要求履歴後の比較を実行する。r13はGPU／モデル正しさPASSに含めず、diagnostic overlayはproductionに取り込まない。

- r14は両条件で診断を完了した。layer 0の12演算およびlayer 1／2末尾は全て一致し、
  最初に観測された差はFull Attentionのlayer 3末尾だった。同期を追加しても要求履歴後の文章崩壊は再現した。
  compact比較は`.local-artifacts/phase83/r14-first-layer-difference.json`に保存した。
  同一binaryの診断対象をlayer 3へ切り替え、projection／preprocess／KV attentionのどこから差が出るか確認する。


- r14のlayer 3全演算比較ではQ/K/V projection、norm scale、preprocessのQ出力は一致し、
  最初の差が`layer.3.causal_attention`で発生した。要求履歴後のhash `332500c9f2b78325`は
  BF16 `0xffc0`（負のquiet NaN）6,144要素のhashと一致する。これはhashからの診断推定であり、
  r15で非有限値の実数と同層KV planeを直接確認する。比較記録は`r14-layer3-comparison.json`。
  診断対象はR9700、Q24／KV4／head dimension 256、M1 replay、KV length 8,193である。


- r15は非有限値を直接数え、要求履歴後の最初のreplay attentionで2,436／6,144要素、起動直後は0／6,144要素だった。
  前者は9 tokenでEOSとなりharnessも失敗、後者は正常128 tokenで成功した。両方ともprocess終了と解放zeroを確認した。
  この不具合はMTP有無の許容される文章差とは区別する。KVの全容量plane hashには未使用tailが含まれるため、
  hash差だけで有効KVの破損を断定しない。r16では同じattentionのQ/K/V/scaleと出力をscratchへ取得し、
  有効8,193行の独立数値oracleで、入力破損とattention内の破損を切り分ける。


- r16のraw dumpを独立MXFP8 decoderで解析し、要求履歴後の有効KVにkey 100要素／value 12要素のInfを確認した。
  生のE4M3 NaN codeではなく、破損scale `0xfd`／`0xfe`によるoverflowだった。Qは全要素finite。
  fresh比較ではkey/valueは最初の2 MiB（rows 0..2047）だけが異なり、rows 2048..8194は一致、
  scale planeは有効prefixのほぼ全域がゼロ／異常値になっていた。再appendしたrow 8192自体は4 planeとも一致した。
  起動直後は全KV／Q／attentionがfiniteで、float64 attention oracleとの最大絶対差0.003876未満だった。
  `replay-kv-dumps/r16-prefix-comparison.json`にraw SHA付きの比較を保存した。未使用tailではなく有効prefixの破損である。
- r17の検証前state exportは未確保の末尾を含む全容量を要求して失敗した（backend status 269）。
  これは追加診断の失敗でありGPU正しさPASSには含めない。r18のscratch adapterはpublished範囲だけをreadbackし、
  image形状維持用の未使用tailはCPU上でゼロを補う。GPU stateを書き換えず、検証前／後／rewind後の有効prefixを比較する。


- r18の段階別dumpで、検証直前の8,192-row prefixは正常、M3 target verify後に既存prefixが破損すると確定した。
  rewind前後および続くM1 replayの前後で、そのprefix bytesは全4 planeとも完全一致した。
  よってrewind／replayそのものをprefix破損の直接原因とする仮説は除外する。
  M3 verifyのhiddenは正常freshと同じ `39ccb1415d23f9a0` のままで、後続層でのページ追加／再利用も調査対象となる。
  数値差は`replay-kv-dumps/799052/stage-comparison.json`へraw SHA付きで保存した。
  公開ABIによる小型検査では8,192-row prefix保持、4つの小buffer確保、M3 append、rewind、M1 appendを個別に調べる。


- 公開ABIの単一stateおよび16-stateページ成長検査はR9700でprefix保持に成功した。
  4つの3,072-byte buffer確保、M3 append、rewind、M1 appendも通り、8,193行attentionはoracle最大1 BF16 ULPだった。
  小型検査では実モデルの破損を再現していない。sourceとログは`kv-prefix-regression/`に保持した。
- r20は最初のverifyだけを診断するため8,192入力／4出力とし、Phase 83の最終8,192／128受入には数えない。
  layer 3 KVの先頭128 bytesはlayer 50末尾まで一致し、keyがFull Attention layer 51、valueがlayer 55で変化した。
  `api-r9700-r20-prefix-watch-sequence/prefix-watch-assessment.json`に記録した。
  後続層の新規KVページと既存ページの所有権／mappingの衝突は仮説であり、該当層の詳細追跡とnative API監査を続ける。


- 同じr20 binaryでlayer 51の各演算を追跡すると、Q/K/V・norm・attention preprocessまでlayer 3 prefixは正常で、
  KV append／causal attentionを挟んだ後に破損した。診断配置によりkey/valueの変化する層は変わるが、
  最初の差をこの2操作まで絞れた。`api-r9700-r20-layer51-sequence/prefix-watch-assessment.json`に記録した。
  native VMMの仮想範囲・live allocation handle・ページ追加前後のprefixを次の診断対象とする。


- 上記小型検査に22,499,948,352 bytesの通常buffer確保（1 GiB以下に分割）を加えてもR9700でprefix保持とoracle最大1 ULPを確認した。
  `kv-prefix-pressure-r2/identity.json`にsource／archive／binary hashと実行条件を保存した。
  buffer全域にモデルpayloadを書き込む検査ではなく、実モデルの配置やアクセス履歴まで再現したとは扱わない。

- r22のnative診断ではlayer 51のKVページ追加直後、append kernelの実行前にlayer 3の既存key先頭128 bytesがゼロ化した。
  layer 55では既存valueにも同じ現象が起き、続くappendで別層のデータへ変化した。readback／同期statusはすべて0、
  仮想アドレスとallocation handleは別だった。`api-r9700-r22-vmm-sequence/vmm-assessment.json`に記録した。
  原因の境界はHIP VMMのcreate／map／accessによるgrowまで絞れたが、HIP内部の具体的な原因は未確定である。
  この8,192／4診断を最終受入には数えない。既存のGPU常駐連続KV経路をgfx1201の通常stateで選ぶ修正をscratchで検証する。
  MXFP8形式、attention kernel、MTPは維持し、最終採用前に要求履歴後の8,192／128と状態保持を確認する。

- r23でgfx1201の通常KV stateを既存`contiguous-resident`へ切り替えると、短文／SSE／cancel／recovery後の
  8,192入力／128出力が成功した。layer 3の既存8,192行はverify前後・rewind後で全4 planeのSHAが一致し、
  最初のreplay attentionは6,144要素すべて有限値だった。文章はRustの所有権についての正常な説明で、BF16品質同等性の証拠ではない。
  MTPは104 draft中76採用、HIP-only、request/workspace cleanup 0、正常終了を確認した。
  logical capacity 8,320のKV commitは281,149,440 bytes。`api-r9700-r23-resident-sequence/resident-assessment.json`に記録した。
  この修正をmainへ反映し、診断overlayのない現行main全体で両GPUの最終検証を続ける。

- resident選択のfocused host回帰検査は1件、KV state関連host検査は28件すべて成功した。
  gfx1201の1／8,191／8,192／8,193／8,320／65,535 capacityと、gfx1030の65,535／65,536／65,537境界を検査した。
  初回の`--exact`による短いfilter名は0件選択だったため証拠にせず、正しいfilterで再実行した。
  `r25-kv-host-tests.log`に28件の結果を保持した。現行mainの両GPU buildも成功し、既存Rust formatting差を修正後のr25を最終検証候補としてビルドする。

- r25の診断なしcurrent-main binaryは両targetでsource-before／after一致とbuild成功を確認した。
  既定設定（opt-inなし）のAPI履歴→8,192／128をR9700／V620-A、CLI同長文をV620-Bで実行中である。
  両APIの短文／SSE／cancel／recoveryは成功した。長文中のVRAMは容量上限近くまで増えるため、完了・解放と
  `r25-live-memory.jsonl`の時系列を確認するまで収容条件をPASSとしない。
- 公開前Clippyで、CLI helper追加時に既存`from_protocol_text`から外れたtoo-many-arguments属性と、
  host testの不要なPathBuf確保を修正した。server／CLIのall-targets Clippy（warningsをerror扱い）とformat確認は成功した。
  生成処理には変更がなく、r25との正確な2箇所の差分とsource SHAを`r25-source-followups.json`へ保存した。
  CI登録情報の検査とpublic-runtime matrixの105 file hash照合も成功した。

- r25 R9700の既定API履歴→8,192／128は完了した。入力コードに沿ったレビュー文、MTP 113候補／70採用、HIP-only、
  cleanup 0、正常終了。長文wall 819.491秒／first content 765.859秒で、prefill単独計時には読み替えない。
  VRAM peak 34,196,402,176／total 34,208,743,424 bytes、GTT増分114,688 bytes、終了後VRAM約61.6 MB。
  収容は確認したが余裕は小さく、別context／modelの収容保証ではない。`api-r9700-r25-default-sequence/assessment.json`へ記録した。
- 累積レビューでgfx1030 BF16 ID91のpublic launcher不具合を検出した。kernelが2D block indexを使う一方、launcherと
  metadataは1D flattened gridであり、M>64の行タイルを書き込めない。既存の直接kernel probeはpublic launcherの証明になっていなかった。
  r25 V620-A APIとV620-B CLIは通信・128出力・cleanupを完了したが、数値受入は無効とし、それぞれ`acceptance-invalidated.json`を保存した。
  1D grid契約を維持してkernelのtile indexを修正し、公開plan／executeを通るM63／64／65／1024の回帰検証後に長文を置き換える。
  ID91を選ばないgfx1201の結果はこの不具合の対象外で、R9700 CLI長文は継続している。
- 同レビューでH3の新kernel／compiler stub inventory漏れと、ID92のRust binding定数漏れも確認した。
  r25の両target native archiveから8 kernel／6 causal stubの追加対象を同定して一覧へ反映し、ID92定数を追加した。
  kernel修正確定後にsource hash inventoryとH3検査を同期する。

- r26でID91の1D public launcherに合わせたtile index修正を完了した。公開prepare／executeを通る
  M63／64／65／1024 × MTPの3 shape、計12ケースがV620で成功した。行依存の独立BF16 oracleで
  64行境界・最終行を含めて照合し、finite／guardも確認した。直接kernel probeだけの証拠を置き換える。
  `id91-public-launch-r26/identity.json`にbinary・source・CTest logのSHAを保存した。
  両targetのr26 release buildは成功し、各buildのsource-before／afterも一致した。
  H3 inventory同期後の検査は86 tests／651 subtestsで成功した（`r26-h3-tests-fixed.log`）。
- R9700のr25既定CLI長文は8,192入力／128出力、MTP 113候補／70採用、HIP-only・fallbackなし・終了code 0で完了した。
  入力Rustコードの所有権・overflow処理に沿う文章を確認した。これはBF16 full-model比品質同等の証明ではない。
  load込みwall 905.485秒、VRAM peak 34,198,695,936 bytes（余裕10,047,488 bytes）、GTT増分90,112 bytes、
  終了後VRAM 61,235,200 bytes。`cli-r9700-r25-default-long/assessment.json`に記録した。
  gfx1030専用ID91修正の影響対象外であり、r26 binaryの結果へ読み替えない。
- r26のV620 CLI／API長文を実行中。APIの短文・SSE・cancel／recoveryは成功した。
  R9700のr26対話モードは1 turnのcommit、session.eof、終了code 0を確認した。
  `cli-r9700-r26-default-interactive/execution.json`とJSONLを保持した。V620の完了前の長文をPASSには含めない。
  現行r26でRustfmt確認も成功した。R9700は通常設定target-onlyの初回単回benchmarkを開始した。
- Phase 82 raw resultの最初のwarmup行と、r26のwarmupなし／単回行をresident後のfresh requestとして比較する。
  `sample_kind`／`sample_index`は報告用で実行経路・seedを変えないことをsourceで確認した。
  model loadは計時外なのでprocess cold-start比較とは呼ばず、反復中央値や安定性能・速度目標達成の証拠にも使わない。
  Phase 82初回値はV620-A prefill 8.303292／decode 8.061141 tok/s、R9700 10.688401／9.717513 tok/s。
  両条件とも同一UUID、fixture、MXFP8、chunk 1024、固定sampling・seed 123、8,192／128で比較する。
  MTPあり測定の設定は`default-mtp-width2-settings.json`のMTP on／width 2のみとし、高速opt-inを混ぜない。
- 現行sourceのMTP関連host証拠を補完した。coreの`mtp` filterは28 PASS／4 ignored、frontendは9 PASS。
  Qwen hidden幅、partial-prefix alias、draft幅・capacity境界、seed／counter／制約、replay後hidden rowの選択を含む。
  外部artifactが必要なignoredをPASSへ数えない。ログは`r26-mtp-contract-host-tests.log`と
  `r26-frontend-mtp-host-tests.log`、既存証拠の索引は`closeout-evidence-index.json`に保存した。
- V620 r26の既定CLI／API履歴→8,192入力／128出力が完了した。両方とも入力に沿うRustコードを生成し、
  MTP 95候補／80採用／15棄却、HIP-only、fallbackなし、正常終了を確認した。APIは全要求cleanupとshutdownがzero。
  API長文wallは1,049.242秒。終了直後のsysfs sampleにはdriver側の解放待ちが残ったが、次のGPU run開始前の
  遅延観測でVRAM／GTTともbaselineへ一致復帰した。CLI VRAM 17,244,160 bytes、API側17,219,584 bytes。
  両runの`assessment.json`と`delayed-post-exit-memory.json`へ記録した。r25の無効化は取り消さない。
  V620-Bの対話モードと、Phase82と同一UUIDのV620-A target-only初回benchmarkを開始した。
- V620-B r26対話はturn commit／session.eof／終了code 0を確認し、両GPUのCLI/API長文と短い対話が揃った。
  r25→r26のbuild source差分は6ファイル、同一454ファイルで、現行sourceはr26全460ファイルと一致する。
  gfx1201で再利用するr25長文証拠の変更範囲を`r25-r26-source-mapping.json`へ記録した。
- R9700 r26 target-only初回比較はprefill 10.715413／decode 9.771964 tok/s、128公開token、HIP-only、
  非有限terminal logit 0、cleanup zeroで完了した。Phase82初回比はそれぞれ+0.253%／+0.560%だが、
  単回差なので改善確定とは扱わない。MTPありの同条件測定を続ける。
- 公開前の既存CI入口でPython compile／static（各228 files）、Markdown local linksを確認した。
  C++ formatに不一致があったためclang-format 18で今回の15ファイルを整形し、formatとhost静的buildが成功した。
  production側の字句token列は一致し、テスト2ファイルのinclude並べ替え／隣接文字列分割も記録した。
  before／after snapshotとhash対応は`r26-format-map.json`および追跡対象の検証記録に保持する。
  r26 binary自体は整形前のままであり、現行sourceとの同一byte主張を意味的対応へ更新した。
  CIのsource hash inventoryを同期し、変更の影響を受けるH3契約検査を再確認中である。
- 整形後H3契約31 tests／463 subtestsが成功した。V620-A target-only初回はprefill 8.305291／decode 8.107713 tok/s、
  R9700 MTPありは10.703184／2.127297 tok/sだった。両方128出力・HIP-only・非有限terminal logit 0・cleanup zero。
  R9700のMTPは128候補／64採用／64棄却で、通常設定ではdecodeが遅い。速度改善はPhase83.5へ残す。
  [初回比較記録](../../../../../history/2026/09/1-10/phase83-first-request-comparison.json)へbaselineと同じfixture SHA・UUIDを記録した。
  V620-AのMTPあり測定を開始し、R9700の既存ユーザーserviceを復帰してhealth待ちとしている。

- 最後のV620-A MTP初回比較は8.257178 prefill／2.254121 decode tok/s、96候補／80採用／16棄却、
  128出力・HIP-only・非有限terminal logit 0・cleanup zeroで完了した。両GPUの機能検証と比較記録が揃った。
  既定MTPは両GPUで遅いため速度改善はPhase83.5へ残し、BF16 full-model品質同等は未証明のままとする。
  R9700の既存ユーザーserviceはhealthz／readyz HTTP200で復帰した。

## 作業境界

- benchmark担当: Phase78 benchmarkとPhase83 fixture／runner。
- MTP担当: core／frontendのMTP実行。API担当と接続契約を共有する。
- API担当: server CLIとproduction backend。
- KV担当: 共通KV native／Rust経路。
- main担当: 統合、prefill候補、GPU／ビルド資源、最終検証・公開。

## 履歴

[履歴](../../../../../history/2026/09/1-10/phase83-mxfp8-fixed-sampling-mtp.md)と
[検証記録](../../../../../history/2026/09/1-10/phase83-integration-evidence.json)を確定し、この計画をarchiveへ移した。
