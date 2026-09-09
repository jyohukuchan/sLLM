# Phase 83: MXFP8 E4 KV・固定sampling・MTP統合の履歴

> 2026-09-09: 実装・実GPU検証と比較記録を完了。公開結果は当該commitのGitHub Checksと完了報告で確認する。

この履歴は[main-plan](../../../../plans/main-plan.md)と[実行計画](../../../../plans/archive/2026/09/1-10/phase83-mxfp8-fixed-sampling-mtp.md)を正本とする。
artifactのパス、SHA、対象、制限は[integration evidence](phase83-integration-evidence.json)に集約する。
ローカルの補助索引は`.local-artifacts/phase83/closeout-evidence-index.json`に保持し、生ログは転載しない。

## 実装した最終動作

- 対象はQwen3.8 27B NVFP4、standard OCP MXFP8 E4 KV、固定GPU sampling（temperature 1、top-p 0.95、top-k 20）、
  BF16重みMTPのsingle GPU／batch 1経路である。CLIとAPIから同じproduction backendを使う。
- MTPはtarget verification、draftの一括処理、採用prefix queue、accept／reject、replay／rollbackを接続した。
  部分確定後はtarget hiddenを再計算して次のdraftへ渡し、`prompt+max_completion_tokens`の残容量でdraft幅を制限する。
  target残り1行は逐次samplingへ切り替える。embeddingとFP8 headはtarget residentと共有する。
- API選択条件が`requires_logits`だけを見て固定GPU selector対応samplingをMTPから外していたため、selectorで処理可能なsamplingを
  MTPへ接続する条件へ修正した。固定profile、host logprobs、greedyを含むcore／frontend契約検査を通した。
- gfx1201の通常KV stateは`contiguous-resident`を選ぶ。要求履歴後のHIP VMM `create/map/access` growでページ追加直後に既存prefixが
  zero化し、scale `0xfd`／`0xfe`由来のoverflowがattentionへ入ることを切分けた。HIP内部の具体的原因は未確定であり、
  VMM growを一般修正したとは主張しない。sliding descriptorと直接C ABIのVMM選択はこの変更の範囲外である。
- MTP要求の専用queue候補は試したが履歴後の崩壊を解消せず、変更と専用testを戻した。shared queueを原因とは断定しない。
  resident共有と要求状態のrollback／replayを現行経路に残し、診断overlayはproductionへ持ち込まない。

## ID91／ID92と公開入口

- gfx1030のID91は、kernelが2D block indexを使う一方でpublic launcher／metadataが1D flattened gridだったため、M>64の行tileを誤る
  不具合を修正した。1D契約を保ったtile indexへ直し、public prepare／executeのM=63／64／65／1024、FC／up／downの12ケースを
  行依存BF16 oracle、finite、guard、64行境界と最終行で確認した（`id91-public-launch-r26/identity.json`）。
- ID92はgfx1030 FP8のM2〜4、K5120／N10240・6144・248320へfused providerを既定接続した。
  公開probeで既存M1 rowwiseの算術との対応と独立BF16 oracle最大ULP差0を確認した。別shape／targetへは広げない。
  最終レビューではRust binding定数の欠落も補い、8 kernelと6 causal stubをH3 inventoryへ追加してmatrix／G2／H3を同期した。
  このr25→r26の定数追加と、Phase83全体でのID92 provider追加を区別する。
- ID91不具合のためV620 r25 API／CLIの数値受入は無効化済みである。通信、128 token、cleanupだけを再利用しない。
  ID91を選ばないgfx1201のR9700 r25 API／CLI長文はこの修正の影響対象外で、r26 binaryへ再ラベルしない。

## 成功証跡の要約

- r26のgfx1030／gfx1201 release buildはsource-before／after一致で成功し、H3は86 tests／651 subtests、public clippyも完了した。
  `candidate-gfx1030-r26-public-launch/identity.json`、`candidate-gfx1201-r26-public-launch/identity.json`、
  `r26-h3-tests-fixed.log`がidentityとログの原典である。
- current MTP host契約はcore 28 PASS／4 ignored、frontend 9 PASS。ignoredは外部artifact不足としてPASSへ数えない。
  原典は`r26-mtp-contract-host-tests.log`と`r26-frontend-mtp-host-tests.log`である。
- Phase81から引き継いだcore host 565 PASS／20 ignored、固定samplingの独立NumPy oracle、先行GPU/API契約は基礎証跡として保持する。
  これはT=1、P=.95、K=20の契約とrollback／recoveryの根拠であり、Qwen3.8 MXFP8のfull-model受入へ直接読み替えない。
- MXFP8 append chainはgfx1030／gfx1201とも長さ31／32／33、独立oracle最大ULP 0、repeat、cancel、cleanupをPASSした。
  これはKV chainの数値証拠であり、full-modelまたはMTP品質の証明ではない。
- V620 r26の既定API／CLIは8,192入力／128出力をPASSした。両方とも入力に沿うRustコード、MTP 95 proposed／80 accepted／15 rejected、
  HIP-only、fallbackなし、正常終了を確認した。APIはcleanup／shutdown 0で、遅延観測後にVRAM／GTTがbaselineへ戻った。
  原典は`api-v620a-r26-default-sequence/assessment.json`と`cli-v620b-r26-default-long/assessment.json`および各delayed-memoryである。
- R9700 r25のAPI／CLI既定長文は8,192／128、MTP 113／70、HIP-only、fallbackなし、cleanup 0で成功し、CLI終了codeは0だった。
  これはID91修正対象外のr25証跡であり、binaryをr26と呼び替えない。r26 interactiveは1 turn committed、session.eof、exit 0である。
- V620 r26 interactiveも1 turn completed、session.eof、正常終了を記録した。短い対話のlifecycle証跡であり、長文の品質同等性や速度比較ではない。
- R9700 r26 target-onlyは単一のresident後初回要求として記録した。prefill 10.7154、decode 9.7720 tok/sという値は安定性能比較へ使わず、
  Phase82の最初のraw warmup要求と同条件の参考比較とする。
- C++15ファイルをclang-format 18で整形し、format／host静的buildが成功した。r26とのbefore／after mappingを検証記録へ含める。
  整形後のH3契約は31 tests／463 subtests、Python compile／staticは各228 files、文書リンクも成功した。

## 通常設定の初回要求比較

同一GPU個体、MXFP8 E4、chunk 1024、固定sampling・seed 123、8,192入力／128出力で比較する。
model resident後のfresh requestであり、load時間は含めない。Phase82は初回warmupのraw行、Phase83はwarmupなし単回である。
sample名は報告用だが、反復中央値・安定性能・速度目標達成の証拠へは読み替えない。
[元結果のSHA・timing・MTP集計](phase83-first-request-comparison.json)を保持する。

| GPU | Phase82 MTPなし prefill/decode | Phase83 MTPなし prefill/decode | Phase83 MTPあり prefill/decode |
| --- | --- | --- | --- |
| V620-A | 8.3033 / 8.0611 | 8.3053 / 8.1077 | 8.2572 / 2.2541 |
| R9700 | 10.6884 / 9.7175 | 10.7154 / 9.7720 | 10.7032 / 2.1273 |

単位はtok/s。R9700のMTPありは128候補／64採用／64棄却、128出力、HIP-only、非有限terminal logit 0、cleanup zeroだった。
V620のMTPありも96候補／80採用／16棄却、128出力、HIP-only、非有限terminal logit 0、cleanup zeroを確認した。
既定MTPは両GPUでdecodeを遅くしており、速度面の実用達成を主張しない。改善はPhase83.5で扱う。

## 採用・保留・不採用の候補

- ID80のstaged attentionとQTILE4は明示opt-inとして残す。QTILE4は非整列境界を含むoracleとrepeatを維持した。
- ID87補償加算は独立oracleで最大BF16 ULP 1以内、finite、repeatを確認したが、大型shapeの比較点が限定され、model-wide品質・速度・既定採用が未確認のためopt-inに残す。
- ID88／ID90小M row-gridはそれぞれgfx1030／gfx1201でM=2／3／4の数値・repeatを確認した。
  既定設定のモデル全体の証拠と混ぜず、速度評価・採用判断をPhase83.5へ残してopt-inを維持する。
- ID89 gfx1201 WMMA補償加算は10ケースでoracle最大ULP 0、repeat一致だったが、旧ID64比約1.05倍で正式baselineでもないためopt-inに残す。
  stage K=64／128はM=63〜65や大shapeで退行し、LDS leading dimension 48、K16 term merge、scaled BF16 operand WMMA、QTILE8は
  改善不足または遅延のためproductionへ採用しない。追加の速度探索はPhase 83.5へ移す。
- ID93 staged32 MXFP8 attentionは両target共通のopt-inとして残す。長さ1023／1024境界、独立oracle、
  workspace／2 dispatch／cleanupの検証範囲とN1分類は[数値変更台帳](../../../../compatibility/numerical-output-changes.md)に記録した。

## Phase 83の方針と制限

- Phase 83は正しい実装、両GPUのCLI／API長文・対話、SSE、cancel／recovery、要求再利用、unload、32 GB級VRAM収容、数値とsamplingの正しさを扱う。
  追加最適化とV620 200／20、R9700 500／25 tok/sの速度目標はPhase 83.5へ分離し、単回のtarget-only値で達成を主張しない。
- MTP有無のtoken列・文章完全一致は要求しない。同一BF16参照への精度劣化を明示した指標・許容差で比較し、sampling差と数値差を区別する。
  token不一致、単一生成例、単一kernel oracleだけでBF16品質同等性を認定しない。要求履歴による文章崩壊や状態破損は許容しない。
- Qwen3.8 BF16 full-modelは約55.56 GB／18 shardで、sLLMの公開入口とmodel lockが未整備、単一32 GB GPUへ収まらない。
  よって現行のBF16 kernel oracle、NVFP4生成文、MXFP8 chainからBF16 full-model品質同等性は導かない。
- 対象外はtools完全対応、vision、全モデル、TP、batchingである。host契約、operator oracle、短い対話、長文生成の各証跡を相互に越えて一般化しない。

## 公開と後続作業

R9700の既存ユーザーserviceは復帰し、`/healthz`と`/readyz`のHTTP200、ready／scheduler acceptingを確認した。
既存構成の復帰であり、新binaryの常駐serviceへの配備は行っていない。
Phase83の実装・検証・比較記録は完了した。公開commitに対するhostと2種類のH3の結果をGitHub Checksで確認する。
初回公開`24af711e`は両H3とhost H1/H2が成功し、H0はbenchmark集計関数のClippy引数数、ローカル専用証拠への4リンク、RMSNorm H3の古いsource hashで失敗した。
当該関数へ既存の同種helperと同じlint属性を追加し、sllm-hipのall-targets Clippyを確認した。
ローカル証拠の所在はリンクから通常のpath表記へ変更し、RMSNorm H3の既存source inventoryを同期した。
推論・計測処理に変更はなく、修正前後のsource hashを検証記録へ保存した。修正commitの公開CIを再確認する。
Phase83.5の追加最適化と速度目標達成、BF16 full-model品質同等性の未証明は区別して保持する。

実行記録: [Phase83計画](../../../../plans/archive/2026/09/1-10/phase83-mxfp8-fixed-sampling-mtp.md)。
