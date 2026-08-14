# Phase 12R: CI portability repairとlocal/remote verification整理

> 状態: completed
> 作成日: 2026-08-15
> Phase割当: Phase 12待機中に実行するremediation subphase。Phase 13〜20は繰り下げない。

## Phase割当の理由

本作業はPhase 7で構築したCI/CDと、Phase 10〜11で追加したFP8/CDNA3 build inputの不整合を修復する。
新しいmodel、kernel、runtime機能ではないため、製品機能のPhase 13を繰り下げず、MI300X実機を待つPhase 12の
local-only remediationとして`Phase 12R`を割り当てる。

Phase 12RはPhase 12の完了、skip、MI300X PASSを意味しない。Phase 12は`ready`のまま維持し、Phase 12Rでは
Hot Aisle VMを作成・起動しない。完了後は既存番号のPhase 13へ進む。

## 目的

GitHub-hosted CIをclone直後のtracked repositoryだけで完結する短いportability gateとして修復し、GPU、model、
性能の正規検証はtrusted local hostへ明確に分離する。localで通る一方GitHub Actionsで失敗する現在の状態を解消し、
同じsuite registryとmatrixからlocal/remoteの実行内容を追跡できるようにする。

CIを廃止しない。GitHub ActionsはRust/C++/Pythonのformat・lint・build、host contract、tiny oracle、文書/schema、
tracked dependency closure、固定ROCm containerでのcompile/linkを担当する。実GPU数値、full model、llama.cpp実体を
使う比較、性能はlocal laneが担当し、CPU/compile結果からGPU PASSを主張しない。

## 開始時点の失敗baseline

対象candidateは`39ffa8eb70063282b623fee714b665ce8de5618a`である。2026-08-15確認時点のGitHub Actions
run `31816408495`ではH1/H2がPASSし、H0だけが次の3分類でFAILした。

1. `h0-cpp-format-static.cpp-format`
   - `native/hip/tests/public_runtime_host_test.cpp`にclang-format 18.1.3との差分がある。
2. `h0-llama-phase5-contract.llama-phase5-host-contracts`
   - Git管理外の`reference/llama.cpp`がないGitHub checkoutで、test collection前の`setUpClass`が失敗した。
3. `h0-rust-dependency-closure.rust-dependency-validator`
   - tracked dependency closureの`workspace_members` graph/fieldが現Cargo workspaceとずれている。

同candidateの通常H3 compile-onlyは`gfx1030`/`gfx1201`ともPASSした。一方、public-runtime H3 run
`31816408464`とRMSNorm H3 run `31816408524`は、FP8 pathが参照するhipBLAS/hipBLASLt symbolをlink commandへ
含めず、undefined symbolでFAILした。semantic RMSNorm G1 run `31816408574`は、push triggerが利用可能な
self-hosted runnerを得られずpendingのままである。

このbaselineは修正対象の固定であり、新しいCI gate、外部review、全GPU matrixを追加する根拠にはしない。

## 固定するverification境界

### GitHub-hosted CIに残すもの

- Rust format、clippy、MSRV、locked dependency closure。
- C++ format、C++17 host build、public header/ABI static contract。
- Python compile/static、H1 host contract、H2 tiny NumPy oracle。
- Markdown/link、JSON/schema/matrix/workflow、license/provenance、tracked tree hygiene。
- 固定ROCm containerでの必要最小限のHIP compile/linkとexact target/code object inspection。

### trusted local hostへ置くもの

- G0〜G4の実GPU identity、allocation、dispatch、数値、model、service、cleanup。
- `reference/llama.cpp`の実checkoutを必要とするsource identity確認と実比較。
- model download、real-weight slice、full generation、性能、VRAM、health観測。
- release/integrationで必要なsemantic/build identityとGPU evidenceの集約。

### 禁止する混同

- local referenceやmodel cacheがある開発機だけでH0のportable PASSを主張しない。
- GitHubのcompile-onlyをGPU runtime、数値、compatibility、性能PASSへ昇格しない。
- self-hosted GPUが不在のpushを無期限pendingにせず、成功や通常skipへも読み替えない。
- duplicated workflowの一方だけがPASSした状態を同じH3 contract全体のPASSと呼ばない。

## スコープ

- H0のformat、llama.cpp contract分離、Rust dependency closure修復。
- H3 core/public-runtime/RMSNormのlink input、matrix、runner、aggregate、workflowの整理。
- push/PR、manual、scheduled、trusted-local GPUのtrigger境界整理。
- suite registry、path-to-suite mapping、dependency manifest、workflow validatorの同期。
- registry/matrixを再利用するlocal host verification entrypointと開発文書。
- current failureを固定したnegative/portability regression test。

次は含めない。

- model execution、sampling、API、kernel数値、provider selectionの意味変更。
- MI300X、V620、R9700の新しいGPU correctness/performance claim。
- H3 required昇格、性能threshold、external-contribution laneの有効化。
- CI serviceの移行、別vendor CI、artifact長期保存基盤、GPU runnerの再image自動化。
- llama.cpp checkout、model、binary、raw log/profileをGitへ追加すること。

## 受入条件

1. GitHub-hosted H0/H1/H2がtracked checkoutだけで収集・実行でき、Git管理外の`reference/`、model、GPU、secretを
   要求しない。
2. C++ formatはCIとlocalで同じclang-format contractを使い、対象tracked C/C++/HIP sourceがPASSする。
3. Phase 5 llama.cpp host contractはtracked fixture/source-lock metadataだけでcollectionでき、実checkoutの存在・commit・
   path/hash検証は明示的なlocal/integration commandでfail-closedに行う。missing referenceをSKIP扱いしない。
4. Rust dependency closureは現workspace graphから決定的に生成・検査でき、member追加・削除、feature/edge drift、Cargo.lock
   不一致を検出する。期待値更新だけで未知edgeを黙認しない。
5. H3のcore、public runtime、RMSNormに必要なHIP、hipBLAS、hipBLASLt link inputが一つの正本から解決され、
   `gfx1030`/`gfx1201`のcompile/link/code object inspectionがPASSする。
6. 重複H3 workflowは責務を統合するか、明示的に異なるrowとして同じmatrixへ収容する。obsoleteなpush workflowを残さず、
   H3はnon-requiredのまま維持する。
7. push/PRで自動実行するのはGitHub-hosted host/compile laneだけとし、self-hosted GPU workflowはmanualまたはtrusted local
   controllerからの明示実行に限定する。runner不在のpushがpendingを残さない。
8. local entrypointはsuite IDを複製せず現registry/matrixを読み、dirty draftを`local-development, immutable=false`、cleanな
   integration候補を対応するidentityとして区別する。
9. current failureを再現するnegative testが、missing local reference、stale dependency graph、missing HIP link dependency、
   self-hosted push trigger再導入を検出する。
10. focused check、H0/H1/H2、canonical H3、1回のintegration review、指摘箇所のfocused re-review、testing/CI方針、
    main plan/historyを同期し、本planをarchiveする。
11. Phase 12Rの変更だけを必要最小限のcommitへ整理し、current GitHub branchへpushする。次Phaseの実装を同じcommitへ
    混ぜず、push後のbranchとworking treeを確認する。

## 実装順序

### P12R-A0: failure inventoryと契約固定

- run `31816408495`、`31816408464`、`31816408524`、`31816408574`のreport/logから失敗分類と実行argvを固定する。
- H0の各失敗をcanonical local commandで再現し、format、portability、manifest driftを別fixtureへ分ける。
- H3 runnerごとのcompile/link source、library order、target、artifact inspection、aggregate責務を比較する。
- workflow eventとrunner labelを一覧化し、push/PR、manual、schedule、localの所有境界を固定する。

### P12R-A1: H0 portability修復

- clang-format 18 contractで対象C++を整形し、format validator自身の対象path/version testを維持する。
- `run_llama_phase5.py`のtracked contract loadとlocal source identity確認を分離する。host testはtracked source-lock fixtureを
  注入し、実checkoutを開く処理は明示local commandだけが呼ぶ。
- missing reference、wrong commit、dirty reference、source-lock不一致のlocal negative testを残す。
- Cargo metadataからworkspace member/edge closureを決定的に取得し、tracked expectation/schemaとvalidatorを同期する。
- H0単体をdirty local modeで実行し、失敗件数0、未知test 0、収集件数非0を確認する。

### P12R-A2: H3 link正本化とworkflow整理

- CMake/native buildとH3 runnerが同じtarget別link dependency定義を使うようにし、FP8を含むpublic runtimeでは
  hipBLAS/hipBLASLtを明示する。
- static library順序、`--no-undefined`相当のfail-closed link、ROCm root、Code Object V6、exact target inspectionを固定する。
- core H3、public-runtime H3、RMSNorm H3の重複compileを比較し、public runtimeへ包含済みの単独RMSNorm rowは
  regression fixtureを移してobsolete workflowを削除する。固有contractが残る場合は同一matrixの明示rowにする。
- `gfx1030`/`gfx1201`のpush向けbounded rowと、weekly/releaseの10 exact target compatibility rowを分ける。
- compile成功をGPU PASSへ昇格しないschema/aggregate negative testを維持する。

### P12R-A3: eventとGPU execution plane整理

- `host-required`はPR、merge group、main pushでGitHub-hosted H0/H1/H2を実行する。
- non-required H3はGitHub-hosted containerでbounded canonical rowを実行し、promotion条件は変更しない。
- semantic G1などself-hosted GPU workflowから通常push triggerを外し、`workflow_dispatch`またはtrusted controllerの明示入力、
  完全commit SHA、default-branch workflowだけを受け付ける。
- Phase 7 daily/weekly/release profileのGPU rowはlocal execution planeを正とし、GitHub側はhost/compile selectionと
  手動control-plane用途だけに限定する。runner不在をPASS/SKIPへ変換しない。
- workflow validatorへ、self-hosted jobのpush trigger禁止とGitHub-hosted required jobの固定を追加する。

### P12R-A4: local parity entrypointと文書

- current suite registry/matrixからH0/H1/H2を選択するlocal wrapperを追加し、suite listを別定義しない。
- dirty draft、clean integration、H3 container、trusted GPU runnerのcommandを明示し、evidence classを混同しない。
- `docs/development/testing.md`とCI・テスト方針へ、GitHub portability lane、local GPU lane、llama source check、
  trigger境界、失敗時の最短再現commandを反映する。
- local wrapperがmodel/GPU testをCPU fallbackで実行しないことをnegative testで確認する。

### P12R-A5: integration確認

- affected Python/manifest/workflow test、Rust workspace、C++ format/static build、Markdown/linkを実行する。
- H0/H1/H2をlocal-development modeでPASSさせ、clean candidateではGitHub `host-required`の3 rowとaggregateをPASSさせる。
- canonical H3 `gfx1030`/`gfx1201`でcore/public-runtimeのcompile/link/code object inspectionとaggregateをPASSさせる。
- GPU runtime意味を変更しない限り広いG1〜G3やfull modelを再実行しない。build/link入力がruntime artifactへ影響する場合だけ、
  対象GPUの短いpreflight/launch smokeを別identityで行い、数値・性能全matrixへ拡張しない。
- integration reviewはportability、dependency closure、link completeness、trigger境界を一回確認し、指摘変更だけを再確認する。

### P12R-A6: closeoutとPhase 13 handoff

- GitHub run ID、candidate SHA/tree、host/H3 aggregate、local command結果、未実施GPU範囲をhistoryへ記録する。
- obsolete workflow、suite、matrix、文書への参照が残っていないことをtracked searchとMarkdown link checkで確認する。
- CI計画、testing文書、main plan、forward queueを同期し、本planをarchiveする。
- Phase 12Rだけをcommitし、upstreamとの差分とGitHub remoteを確認してforceなしでpushする。各後続Phaseも同じ
  Phase単位のcommit/push境界を使用する。
- Phase 13へはgreenなhost portability baselineとlocal verification entrypointだけを渡し、CI固有型やworkflow都合を
  model-neutral execution設計へ混ぜない。

## Verification lane

| lane | 内容 | Phase 12Rでの扱い |
| --- | --- | --- |
| H0 | format/lint/docs/schema/workflow/dependency/portability | 各affected work unit、最終clean candidate |
| H1 | host contract | A5で一回、H1 source変更時はfocused |
| H2 | tiny NumPy oracle | A5で一回、数値意味は変更しない |
| H3 | fixed ROCm compile/link/code object | link/matrix変更ごとにcanonical 2 target |
| G0/G1 | 実GPU preflight/短いlaunch | artifact link入力が変わり必要な場合だけfocused |
| G2/G3/P | model slice/full model/performance | 通常は実行しない。Phase 12Rのclaim外 |

## Rollbackと再計画

- host testを通すために`reference/llama.cpp`をGitへ追加せず、tracked fixtureとlocal source verificationの分離を戻せる単位で行う。
- H3統合で固有のABI/code object検査が失われる場合はworkflow削除を止め、同一matrixの独立rowとして残す。
- GitHub-hosted runnerでROCm containerのbounded compileが15分を超える場合は、timeoutを伸ばす前に重複compile、artifact、
  target選択を削減する。
- self-hosted GPU workflowをmanual化してもlocal runner commandとfail-closed reportは削除しない。
- 同じwork unitの2回reject、review時間が実装時間超過、1時間以上の機能進捗停止、検証・文書が30%超、
  見積り1.5倍超、gate/受入条件変更のいずれかで追加review・検証を止め、ユーザーへ報告して再計画する。

## 完了記録

- P12R-A0〜A6は2026-08-15に完了した。tracked checkoutだけのH0 contract、clang-format 18、Rust dependency
  closure、HIP link closure、workflow trigger境界、registry-driven local entrypointを同期した。
- RMSNorm H3はpublic-runtimeと同じlink closureを再利用する一方、wave32/wave64の両registrationと専用artifact schemaを
  検査する固有rowであるため削除せず、push/PR自動起動を外したmanual regressionとして残した。canonical automated H3は
  public-runtime rowである。
- local-development H0/H1/H2とaggregate、clean integration snapshot上のcore/public-runtime/RMSNorm H3両targetとaggregateを
  PASSした。compile-onlyをGPU実行または数値PASSへ昇格していない。
- integration reviewの指摘は、llama conversion identityの部分比較による契約弱化1件であり、tracked metadataだけを使う
  完全一致検査へ修正してfocused re-reviewを完了した。
- GitHub clean candidateの最終run IDとPhase commit SHAはpush後の履歴追記対象とし、Phase 13へはCI型ではなくgreenな
  host portability baselineとlocal entrypointだけを渡す。

[対応する履歴](../../../../../history/2026/08/11-20/phase12r-ci-portability-repair.md)
