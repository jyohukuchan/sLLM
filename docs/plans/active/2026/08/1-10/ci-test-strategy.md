# CI・テスト方針策定計画

## 目的

開発初期から細かな不具合を検出しつつ、GPUで行うべき処理をCPUで無理に再現して長時間を費やす運用を禁止する。CPUだけで確認できる契約と、実GPUでしか確認できない事実を分離し、各変更に必要な最小の検証を短時間で実行する。

この計画はCI workflowそのものの実装前に、テスト階層、実行時間予算、GPU runnerの安全境界、正しさの証拠、実装順序を固定する正本である。

## 調査結果

### ローカル参照実装

- `reference/llama.cpp`、`reference/vLLM`、`reference/SGLang`、`reference/AMD-ATOM`、`reference/TensorRT-LLM`、`reference/LMDeploy`、`reference/KTransformers`には、`docs/references/source-lock.md` に固定したofficial origin・version・完全SHAのsourceが配置済みである。7件ともshallow、detached、cleanであり、6件はrecursive submodule statusが空、KTransformersは4 gitlinkが全て未初期化で各submodule worktreeが空である。`reference/` は引き続きignore・未追跡である。
- 取得sourceのlicense、path、特殊なLFS/vocabulary fixture、KTransformersの未初期化gitlinkの事実はsource-lock manifestに固定した。追加調査対象からはLMDeployとKTransformersだけを正式採用し、MLC LLM、Candle、CTranslate2、OpenVINO GenAI、ONNX Runtime GenAI、TGIは今回未採用で、cloneも今後の採用予定もない。
- 7件の固定exact revisionを一次sourceとしてCI/testを再調査した。段階化、明示登録、決定的sharding、per-test timeout、preflight、artifact再利用、isolated test、warmupとmetric記録を採用する。
- 暗黙skip、0件収集の成功、required testの`continue-on-error`またはsoft-fail、可変外部artifact/model、root/privileged runner、外部live統計への実行時依存は採用しない。
- source別の完全SHA、主要根拠、採否は[exact-revision調査](../../../../../references/ci-test-exact-revision-review.md)に記録する。これらは設計上の参考であり、uLLMの対応実績または正しさの証拠にはしない。

### CI運用

- GitHubは、公開repositoryのforkがpull requestを通じてself-hosted runner上で危険なコードを実行できるため、self-hosted runnerをprivate repositoryだけで使うことを推奨している。
- 従って、公開repositoryの通常の`pull_request`からself-hosted AMD GPU runnerを直接選択しない。labelやenvironment approvalだけをsecurity boundaryとみなさない。
- GPU実行はdefault branchに置いた信頼済みworkflowから行い、PRのcommitを実行する場合は、review後の明示操作と、jobごとに破棄または再image化できる隔離runnerを必須とする。
- ROCmのdevice可視化環境変数は通常のdevice選択には使えるが、実行コード自身が変更できるため、信頼できないコードに対するsecurity isolationには使わない。

## 基本原則

1. GPU kernel、GPU-scale GEMM/attention、model推論、GPU性能をCPU emulationで証明しない。
2. CPU CIはhost側の意味論、境界検証、error処理、build、極小oracleだけを担当する。
3. compile成功、GPU上での実行成功、数値一致、full model動作、性能を別々の証拠として記録する。
4. GPUがない場合にCPU fallbackへ切り替えてGPU testを成功扱いしない。
5. 新しいopまたはkernelには、同じ変更でhost contract、独立oracle、対象GPU testを追加する。
6. 全組み合わせの直積を回さず、変更影響と代表tupleに基づく明示的なmatrixを使う。
7. 2の冪だけでなく、空、最小、奇数、素数、非整列、tile・vector・chunk境界の前後を必ず含める。
8. timeout、crash、hang、数値不一致、割り当て済みGPUの不在、test未収集を成功または通常skipに変換しない。
9. 性能値をCPU実行から推定せず、同じ実GPU tupleの履歴と比較する。
10. CIを後付けにせず、repository skeletonと最初のruntime contractから導入する。

## CPU CIが禁止する処理

通常のCPU CIは次を行ってはならない。

- Qwen3.5-4B等のfull weightのdownloadまたはload。
- full modelのforward、generation、quantization、形式変換。
- HIP kernelまたはGPU-scale attention/GEMMのCPU emulation。
- production相当shapeを使ったCPU benchmark。
- llama.cppとのfull model CPU比較。
- GPU test失敗時のCPU fallback。
- GPUを待つretryや無期限wait。
- 本番shapeに似せることだけを目的とした大容量tensor allocation。

fake backendはscheduler、execution plan、resource lifetime、error propagation等のcontrol-plane contractに限って使用できる。数値kernelの正しさやGPU対応の証拠にはしない。

## テスト階層

| ID | 階層 | 実行環境 | 検証対象 | 初期予算 |
| --- | --- | --- | --- | --- |
| H0 | 静的検証 | GitHub-hosted CPU | format、lint、Markdown/link、schema、license、workflow構文、tracked tree hygiene | hard timeout 8分/job |
| H1 | host contract | GitHub-hosted CPU | Rustのparser、model lock、scheduler、sampling、descriptor、layout、fake backend、C ABIのerror mapping、API validation | hard timeout 10分/job、通常1秒未満/test |
| H2 | tiny oracle | GitHub-hosted CPU | Python+NumPyによる極小op、dtype変換、KV indexing、sampling helperの独立参照 | hard timeout 8分/job、通常2秒未満/case |
| H3 | HIP compile-only | GPUなし、固定ROCm toolchain | C++/HIP構文、target別codegen、CMake/Cargo integration、ABI static assertion、binding差分 | 5分/target、15分/job |
| G0 | runner preflight | 対象AMD GPU | exact target、driver/runtime/library、device health、binary key、capability probe | 5分/tuple |
| G1 | GPU kernel・ABI | 対象AMD GPU | allocator、queue/event、非同期lifetime、C ABI、capability拒否、個別kernel | 10分/suite、30秒/test |
| G2 | model slice | 対象AMD GPU | embedding、norm、attention、MLP、KV、logits等の実weight sliceとsynthetic tiny model | 10分/suite |
| G3 | end-to-end | 対象AMD GPU | 固定Qwen3.5-4B、weight/activation BF16、KV FP16、CLI、prefill/decode、tokenizer/template、API profile | 30分/job、load後3分/request |
| G4 | compatibility | 対応候補tuple | exact/generic code object、codegen feature、runtime library path、診断manifest | 10分/tuple |
| P0 | performance smoke | 対象AMD GPU | kernel latency、TTFT、TPOT、token/s、peak VRAMの短い観測 | 15分/cell |
| P1 | performance regression | 対象AMD GPU | 履歴baseline、llama.cpp同条件比較、stress | weekly/release内で最大90/180分 |

予算はjobの上限であり、上限まで使うことを目標にしない。Phase 1ではH0、H1、H2を独立したPR required rowとして並列実行し、2分上限の集約を含むrequired workflow全体をp95 10分以内、hard上限15分とする。setup、cache restore、artifact upload、集約をwall timeに含める。予算超過時はtest分割、依存削減、cacheまたはbuild構成の改善を優先し、timeout延長だけで解決しない。

Phase 2でH3をnon-requiredとして追加する。20回以上かつ7日以上の連続観測で、期待rowが全て`PASS`、`FAIL`/`SKIP`/`QUARANTINED`/cancel/schema errorが0、artifact hashが全て一致し、container pullを含むp95が12分以下、最大15分以下、unexpected `INFRA_ERROR`が0、missing resultが0の全条件を満たした後だけrequired昇格をreviewする。

G3 smokeはmodelを事前配置したrunnerで、最大5 request、入力token長`1`、`7`、`255`、`256`、`257`、各出力8 tokenを初期caseとする。nightlyでは入力token長`1`、`7`、`255`、`256`、`257`、`513`、各出力32 tokenまでを初期caseとする。model cache miss時にjob内でdownloadして時間上限を回避せず、preflightの`INFRA_ERROR`とする。P1の90分はweeklyの1 tuple、180分はreleaseの1 tupleあたりのworkflow上限とし、case数、tuple数、setup時間をreportへ記録する。

## CPU oracleの上限

- NumPy oracleは演算の意味を確認できる最小shapeを使う。
- PRの既定上限を1 case内の全tensor合計256K elements、約4M multiply-addsまたは1600万scalar operationsとする。attention等はshapeだけでなく二次的な演算量を見積もる。
- PRでは各opについて必須境界caseと、seed固定の追加caseを最大8件実行する。
- nightly相当のhost property testでも追加caseは最大64件とし、seedと完全shapeを記録する。
- 上限を超える必要があるtestは理由を記録し、`slow`へ分類して通常PRから外す。CPUでGPU規模へ拡大することを解決策にしない。
- H2 test processは最大RSS 4 GiB、fixture合計64 MiB、wall time 5分を内側上限としてcgroupまたは同等の外部制限で強制する。jobのhard timeout 8分にはsetup、cache restore、report生成、artifact uploadを含める。
- H1のfake backendはmetadataだけを扱い、16 MiBを超えるtensor payloadのmaterializeと数値kernelの実行を拒否する。
- dependency取得とtest実行を分ける。test processではnetworkを無効化し、model cacheをmountせず、full model名・lock・weightへのアクセス要求を即時失敗させる。
- JAXはNumPyで計算量または実装上の限界が確認された場合だけ使用する。`JAX_PLATFORMS=cpu`を設定し、実行時にもCPU backendだけであることをassertして、GPU oracleとして暗黙に使わない。

## shape・境界値方針

固定値だけでなく、実装の境界からcaseを生成する。

- contractが許す場合の`0`、最小値`1`、小さい奇数`3`、`7`。
- tile、vector width、wave、block、chunk、page等の各境界`B`について`B-1`、`B`、`B+1`。
- 非整列・素数の代表として`17`、`37`、`73`等。
- token/chunk境界の代表として`255`、`256`、`257`、`511`、`512`、`513`、`1023`、`1024`、`1025`。
- batchは少なくとも`1`、`2`、`3`、`7`を候補とする。ただしMVPのend-to-endはbatch=1に限定する。
- contiguousだけでなく、contractが許すpadded stride、non-contiguous view、byte offset、unaligned rejectionを含める。
- `(M, N, K)`等は全直積ではなくpairwiseと、各kernelの危険な組み合わせを使う。

`0`、最大context、最大batch、stride/alignmentの合法範囲は各descriptor contractで明示する。未定義値をtestだけで既成事実にしない。

## 数値正しさ

- ID、shape、stride、status、serialization、model lock fingerprint、API errorはexact matchとする。
- dtype変換は、定義した丸め・飽和・NaN/Inf規則に従ってbit exactを基本とする。
- 浮動小数点比較は`abs(actual-reference) <= atol + rtol * abs(reference)`を使えるが、`atol`と`rtol`はop、入力範囲、accumulation dtype、出力dtypeごとに登録する。
- `allclose`の真偽だけでなく、最大絶対誤差、最大相対誤差、outlier数、NaN/Inf分類を記録する。
- tolerance未登録の浮動小数点testはfail closedとし、全op共通の緩い既定値を置かない。
- tolerance変更には失敗原因、reference contract、対象GPU tuple、旧値と新値の根拠を必要とする。testを通す目的だけの緩和は禁止する。
- model sliceとend-to-endでは、用途に応じてgreedy token一致、top-1一致率、KLD、BF16比誤差を併記する。最終閾値はQwen3.5-4Bの固定revisionとbaseline kernelで分布を測定後に決める。

NumPy oracleは入力dtype、accumulation dtype、丸め位置、出力dtypeを明示し、最適化HIP実装と同じ制御構造を写さない。

## suite登録とmarker

- tier markerは`tier_h0`〜`tier_h3`、`tier_g0`〜`tier_g4`、`tier_p0`、`tier_p1`とする。
- 直交属性は`requires_gpu`、`requires_model`、`slow`、`network`、`quarantined`とする。
- markerやrunner labelだけをtest選択またはsecurity boundaryにせず、`ci/matrix/suites-v1.json`のversioned suite registryと`ci/matrix/path-to-suite-v1.json`へ明示登録する。
- 未登録test、未知marker、期待収集件数0はH0でfailureとする。required rowはregistryとpath-to-suite manifestから解決した期待suiteを全て収集したことをassertする。
- 推定時間によるshardingは、同じregistry revisionと入力から同じ分割を生成する決定的方式に限定する。live外部統計をrequired workflowの入力にしない。

## 結果状態

| 状態 | 意味 |
| --- | --- |
| `PASS` | 宣言した環境でtestを実行し、contractを満たした |
| `FAIL` | assertion、数値、crash、hang、timeout等でcontractを満たさなかった |
| `SKIP` | matrix manifestで明示した非該当条件。検証証拠にはならない |
| `INFRA_ERROR` | runner、driver、device、cache等の実行基盤が不正。required jobは失敗扱い |
| `QUARANTINED` | 既知failureを隔離中。promotion evidenceにはならない |

- 割り当て済みtupleでGPUまたはtoolchainが見つからない場合は`SKIP`ではなく`INFRA_ERROR`とする。
- `supported` tupleの必須testが非該当になる設計は認めない。
- `QUARANTINED`にはissue、owner、導入日、期限、対象tupleを必須とし、広いretryを行わない。
- retryはnetwork downloadやrunner registration等の一時的infra処理に限定し、数値不一致、crash、hangをretryでpassへ変えない。
- test filter適用後の収集件数をassertし、想定外の`0 tests collected`をfailにする。

GPU test開始前のrunner、driver、device、cache検査失敗は`INFRA_ERROR`とする。test開始後のkernel timeout、device fault、crashはcode起因の可能性があるため`FAIL`とし、同時にrunnerをquarantineする。

result JSONとGitHub job conclusionは次のように対応させる。

- `PASS`: exit 0。
- `FAIL`: exit 1。
- `INFRA_ERROR`: exit 2。required checkはfailure。
- schema不正、未知state、matrix row欠落、report欠落: exit 3でharness failure。
- required rowのcancel、artifact upload失敗、集約前のjob欠落は集約jobをfailureにする。
- required jobでは`continue-on-error`を禁止する。
- `QUARANTINED`はissue、owner、期限を持つ非required専用workflowへ隔離し、required集約checkとpromotion evidenceへ入力しない。そのworkflowに限ってfailure継続を許可する。
- required集約jobは全期待rowのresult JSON、終了コード、report artifactをfail-closedで照合する。

Phase 1で`ci/schema/test-result-v1.schema.json`を作成し、各rowのresultを検証する。v1では少なくとも次を必須概念とする。

- `schema_version`、result ID、suite ID、tier、state、required属性。
- run ID/attempt、reviewed/tested/workflowの完全SHA、Git tree OID。
- matrix manifest SHA-256、matrix row ID、tuple digest。
- command、toolchain、artifact content/manifestのSHA-256。
- 開始・終了時刻、duration、収集・選択・pass・fail・skip件数、seed、case list、diagnostic。
- GPU rowではGPU UUID/BDF/exact target、selected backend、dispatch ID/count、fallbackの許可・使用、code object metadata。

required rowでは`SKIP`と`QUARANTINED`を禁止する。field名と型の詳細はschema実装時に固定するが、上記の意味を削除または弱めない。

GitHub Actions実装では、各required jobのreport生成とupload、および集約jobを`if: ${{ always() }}`で起動する。集約jobは`needs`全体を入力にし、次を検証する。

- `needs.<job>.result`が`success`以外、未知、欠落の場合はfailure。
- `run_id`、`run_attempt`、`reviewed_sha`、`tested_sha`、`workflow_sha`、`matrix_manifest_sha256`が現在のrunと一致すること。
- `created_at`と`finished_at`が現在のrun期間内であり、将来時刻または前回runのstale reportでないこと。
- 期待matrix rowごとにreportがちょうど1件あり、missing、duplicate、unknown rowがないこと。
- required rowが全て`PASS`であり、unknown state、`SKIP`、`QUARANTINED`、`INFRA_ERROR`、`FAIL`、cancelがないこと。
- artifactが現在のrun/attemptから取得でき、content hashとreport hashが一致すること。

集約job `host-required`を唯一の安定したrequired check名としてbranch protectionへ登録する。失敗時artifact uploadも`always()`で試みるが、required evidenceのupload失敗自体をfailureとする。

## CI eventとrunner

| Event | CPU | AMD GPU | 用途 |
| --- | --- | --- | --- |
| `pull_request` / `merge_group` | Phase 1はH0〜H2を必須。Phase 2はH3をnon-requiredで追加し、昇格条件を満たした後だけH0〜H3を必須化 | 直接使用しない | forkを含む高速・安全な検証 |
| maintainerによる信頼済み実行 | 必要に応じ実行 | G0〜G3 | review済み完全SHAのmerge前確認。初期は専用local host、将来は隔離・使い捨てrunner |
| protected `main` push | H0〜H2 requiredとH3 non-required。昇格後はH0〜H3を`host-required`へ含める | canonical tupleでG0〜G2、変更によりG3 | merge直後のGPU smoke |
| daily schedule | smoke | 代表1 tuple | health、flaky、短い性能観測 |
| weekly schedule | full host | 利用可能な明示tuple一覧 | broad correctness、compatibility、性能履歴 |
| protected release | full host | release対象の全明示tuple | release evidence。途中cancelしない |

- public forkのPR codeを永続self-hosted runnerで実行しない。
- `pull_request_target`はmetadata/label等に限定し、PR headをcheckoutして実行しない。
- GPU workflowはdefault branch上の定義だけを使う。
- runnerは可能ならVM passthrough、次にdevice ACLを限定したcontainer、最後に専用bare metalの順で隔離する。
- visibility環境変数をsecurity boundaryにしない。
- GPU runnerは原則1 jobごとのephemeral/JIT登録とし、終了後に外部controllerがprocess確認、診断保存、rebootまたはreimageを行う。
- GPU resetはjob内の任意コードへ許可せず、全process停止後にhost controllerだけが行う。
- runner groupをrepositoryとGPU workflowへ限定し、GitHub environment approval、branch protection、最小`GITHUB_TOKEN`権限を併用する。
- third-party Actionsは完全commit SHAで固定する。

初期GPU evidenceは専用local host上の`gfx1030` 1台と`gfx1201` 1台をcanonical runtime rowとし、相互干渉を避けて直列実行する。2台目の`gfx1030`はspareまたはnightly再現確認用とし、同一changeの必須rowを増やさない。UUID/BDFはG0実装時にtuple manifestへ固定する。

GitHub self-hosted GPU基盤が完成するまでは、maintainerがlocalで作成または完全にreviewしたtrusted project commitだけを、確認したcommandから40桁SHA指定で実行し、同一SHAのevidenceをfail-closed集約へ入力する。この暫定経路ではfork PR head、外部提供binary、未review scriptを実行せず、secretまたはcontroller credentialを注入しない。実行後はprocess残留、device health、artifact hashを確認し、異常時はhostをquarantineする。self-hosted化後はdefault-branch control workflow、ephemeral JIT registration、専用非特権runner user、secret・`sudo`・Docker socketなし、job後のprocess検査とreboot/reimage/quarantineを必須とする。public fork PRのGPU pre-merge実行はこの隔離基盤が完成するまで行わない。

GPU control workflowはprotected default branchのimmutable revisionを信頼元とし、PR側のworkflow定義を使用しない。実行対象codeはmaintainerがreview済みとして記録した完全commit SHAだけを受け付け、branch名や可変tagを入力にしない。許可済みSHA、reviewer、元PR、workflow revisionをcontrol-plane artifactへ記録する。

PR由来のproject scriptとbinaryは隔離runner内の非特権userで信頼できないcodeとして実行する。jobへsecret、host controller credential、cloud credential、Docker socket、`sudo`を渡さない。modelはhost controllerが事前検証したread-only mountから提供し、jobから外部model storageへ直接認証させない。runnerは成功・失敗にかかわらずjob後に破棄または再image化する。

GPU hard gateは変更が実際に触れるscopeへ適用し、未実装の後続tierをbootstrap変更へ循環的に要求しない。一方、適用対象となったtierは同じimmutable candidateで省略しない。

- schema、matrix、runnerの非実行contractだけの変更はH0〜H2とnegative self-testを必須とする。GPU/runtime behaviorも変える場合は下記の該当gateを追加する。
- H3 toolchain、compile-only、artifact metadataだけの変更はH0〜H3を必須とし、compile結果をGPU実行evidenceへ昇格させない。
- trusted local runnerとG0 preflightの変更はH0〜H3、host側negative test、canonical deviceのG0、実行前後healthを必須とする。
- model-free native HIP実行、C ABI、lifetime、allocator、queue/event、fallback、dispatchに影響する変更はH0〜H3とcanonical `gfx1030`/`gfx1201`のG0/G1を必須とする。
- model pathへ影響する変更からG2、互換性の昇格・表記変更からG4、性能または実運用dispatchへ影響する変更からP0を必須とする。

`tested_sha`はreview済みの完全40桁`reviewed_sha`と一致しなければならず、branch、tag、別commit、merge後のSHA、古いartifactを代用しない。該当scopeのrunnerまたはevidence経路が未整備なら、その機能変更をprotected mainへmergeしない。

G1、G2、P0 reportには少なくとも、report ID、run ID/attempt、reviewed/tested/workflow SHA、matrix row ID、tuple digest、selected backend、GPU UUID/BDF/exact target、dispatch ID/count、CPU fallbackの許可・使用有無、artifact content/manifest SHA-256、target/codegen feature、state、開始・終了時刻を含める。GPU PASSでは`selected_backend=hip`、GPU dispatch数1以上、CPU fallback未使用、artifact hash一致を必須とする。

## matrix設計

GPU matrixを独立軸の直積で生成せず、検証する完全tupleを`include`形式の明示rowとして管理する。各rowは少なくとも次を含む。

- Ubuntu release/point release、kernel、amdgpu driver、`amdhsa` execution ABI。
- ROCm build/runtime release、compiler、解決済みlibrary path。
- GPU product、UUID/BDF、exact `gcnArchName`。
- exact/generic target、generic processor version、code object version。
- `xnack`と`sramecc`の`unsupported`/`any`/`off`/`on`、codegen wave size、runtime device wave size。
- capability profileと対象kernel path。
- model lock fingerprint、weight/activation/KV dtype。
- test tier、case set、seed、timeout。

各runはversioned JSON manifestとtest reportをartifactとして保存する。runner labelはrouting用であり、preflightで取得した値を事実の正本とする。

Phase 1では`ci/schema/compatibility-tuple-v1.schema.json`、`ci/matrix/suites-v1.json`、`ci/matrix/host-v1.json`、`ci/matrix/path-to-suite-v1.json`を正本pathとして作成する。schemaとmanifestを可変外部dataから実行時生成しない。

## artifact・cache・保持期間

- Qwen3.5-4B weightをGitHub Actions artifactへuploadしない。
- GPU hostのread-only cacheまたは外部immutable storageを使用し、model lockのSHA-256で毎回検証する。
- fork PRまたはPR codeを実行するGPU jobからtrusted cacheへwriteさせず、cacheを署名済みartifactとみなさない。
- PRはtest summary、seed、失敗時の最小diagnosticを7日保持する。
- `main`/nightlyはtuple manifest、test report、短いmetric、diagnosticを30日保持する。
- release evidenceは90日保持し、長期保存が必要ならGitHub artifact以外へ移す。
- token、secret、model weight、不要なmemory dumpをartifactへ含めない。

## 性能テスト

- 初期は性能を観測値として保存し、履歴が安定するまで通常PRのhard gateにしない。
- 同じexact GPU、ROCm/compiler、model lock、dtype、shape、batch、concurrency、warmup、反復回数だけを比較する。
- TTFT、TPOT、token/s、peak VRAMを保存する。
- medianと分散またはrobustな分位点を保存し、単発値で回帰判定しない。
- llama.cpp比較は固定commit、同じmodel revision、入力長、出力長、dtype、GPU targetで実GPU上だけで行う。
- kernel/runtime pathに触れる変更では短いP0をGPU sanity evidenceとして必須にするが、性能不安定だけでcorrectness結果を上書きしない。
- hard thresholdは十分なbaseline履歴、runner noise、再現率を確認後、metricとtupleごとに設定する。

### `B-1/B/B+1` performance-cliff sanity

- 性能に影響する各kernel/dispatch境界`B`について、合法な`B-1`、`B`、`B+1`を同じP0 case setで測定する。
- 三点は同じreviewed SHA、tuple、model lock、dtype、batch、concurrency、warmup、反復数、runner条件で実行する。
- 各点のselected backend、dispatch ID/count、fallback使用、artifact hash、median、robust spreadを記録する。
- 合法な点の欠落、重複、stale/cancel、非GPU実行、測定値不正、CPU fallback、未許可dispatchはfailureとする。
- 境界でdispatch IDが変わること自体はfailureにせず、versioned dispatch manifestで許可された選択か確認する。
- baseline分布がない段階では全GPU共通の倍率thresholdを発明しない。tripletの完全な実測証拠と`performance_sanity_disposition`を必須にする。threshold未承認時は`review_required`とし、reviewer、理由、日時を伴う承認なしにPASS集約またはmergeしない。
- 履歴が蓄積した後、metric、tuple、dispatch ID、case setごとにversioned thresholdを承認する。

## 変更影響による選択

最初からpath-to-suite mappingを管理する。

- Rust frontend/model lock/APIだけの変更: H0〜H2。GPU contractに影響する場合だけG2/G3を追加。
- H3 toolchain、compile-only、artifact metadataだけの変更: H0〜H3。GPU実行evidenceは要求せず、実行済みとも表記しない。
- trusted local runner、G0 preflightだけの変更: H0〜H3、host negative test、同一reviewed SHAのcanonical G0、実行前後health。
- runtime descriptor、backend/capability/dispatch、C ABI、lifetime、allocator、queue/event変更: H0〜H3、同一reviewed SHAのcanonical G0/G1。model pathへ影響するときだけG2、性能または実運用dispatchへ影響するときだけP0を追加。
- HIP kernel、fallback、native build変更: H0〜H3、同一reviewed SHAと対象tupleのG0/G1。semantic numerical opは独立oracleと境界case、model pathはG2、性能pathはP0を追加。
- tokenizer/chat template/model integration変更: H0〜H2、G2/G3。
- build、ROCm、target、codegen変更: compile-only scopeならH0〜H3。runtime artifactへ適用する場合は同一reviewed SHAのG0/G1、互換性昇格時はG4、model/performanceへ影響するときだけG2/P0を追加。
- quantization/dtype変更: H1〜H3、対象GPUのG1/G2、数値評価、P0。
- scheduler/batching変更: H1、fake backendのcontrol-plane test、実GPUのG2/G3。CPUでGPU workloadを再現しない。

GPUに影響する変更は、実GPU evidenceが得られるまで「compile済み」または「host contract確認済み」とだけ表記し、「GPU verified」へ昇格させない。

## 実装段階

### Phase 0: 方針確定

- 初期GPU evidenceの所有形態、canonical `gfx1030`/`gfx1201` row、直列実行、将来のself-hosted隔離方針を確定した。
- test result/tuple schemaの必須概念、marker、正本path、時間予算を確定した。
- source-lock manifestの完全SHAを対象に、llama.cpp、vLLM、SGLang、ATOM、TensorRT-LLM、LMDeploy、KTransformersのCI/testを一次sourceとして再調査し、採否を方針へ反映した。
- license、provenance、model lock、CI・test、repository hygiene、credential方針をgovernance baselineとして機能codeより先にcommit・pushする。

### Phase 1: repository skeletonとCPU CI

- `tests/contracts`、`tests/reference`、`tests/fixtures`、`tests/api`を用意する。
- `ci/schema/test-result-v1.schema.json`、`ci/schema/compatibility-tuple-v1.schema.json`、`ci/matrix/suites-v1.json`、`ci/matrix/host-v1.json`、`ci/matrix/path-to-suite-v1.json`と共通runnerを置く。
- Rust format/lint/test、C++ format/static check、Python test、Markdown/schema検証を追加する。
- [repository hygiene方針](../../../../../development/repository-hygiene.md)に従うtracked tree H0検査とlocal hygiene commandを追加する。
- H0〜H2を並列PR required rowとし、`host-required`へfail-closed集約する。H3はまだrequiredにしない。
- timeout、収集件数、seed、case timingをtest harnessから必ず出力する。

実施状況: **完了**。

- Rust workspace、CMake C++17 static host stub、versioned C ABI、checked-in bindingsを実装した。host stubはHIP未構築を明示し、CPU fallbackまたはGPU evidenceにしない。
- 2つのschema、suite/host/path manifest、共通runner、fail-closed aggregator、tracked/local hygiene commandを正本pathへ実装した。
- H0、H1、H2を独立rowとして実行し、全row `PASS`と`host-required`集約`PASS`を確認した。
- fail-closed self-testで、意図的なformat/test/schema/0件、missing/duplicate/unknown/stale/hash/identity不一致、non-success needs、禁止tracked pathを拒否した。
- 追加監査で判明したactual selected 0件、dirty/mismatched identity、network未隔離、resource/output/fixture超過、fixture mapping driftもfail-closed対象へ追加した。
- Python 3.12/Linux x86_64 dependencyをtransitive dependencyとSHA-256まで固定し、Rust 1.97.1/MSRV 1.85.0を各commandで明示選択した。
- required commandを外部routeのないnetwork namespaceで実行し、H2の4 GiB address-space limit、row-wide timeout、max RSS、fixture/output上限をmachine-readable resultへ記録・検証した。
- 恒久的な実行入口、出力、CPU-only境界は[host build and test entry points](../../../../../development/testing.md)を正とする。

### Phase 2: HIP compile-only

- ROCm 7.14.0固定toolchainでH3を追加する。
- prebuilt imageをdigestで固定し、`ROCM_PATH`、`amdclang++`、LLVM、headers、device libraries、CMake packageが同じROCm 7.14.0 rootから解決されたことをjob冒頭で検証する。CI中にROCmを都度installしない。
- PRではexact `gfx1030`と`gfx1201`を独立rowでcompileし、nightly/releaseでは`gfx1030`〜`gfx1036`、`gfx1200`、`gfx1201`、将来の`gfx942`を各明示rowとしてcompileする。`gfx1200` compileを実機互換性の証拠にしない。
- exact/generic targetとcodegen featureを混ぜず、artifact metadataを検証する。
- compile-only結果を実機互換性または性能の証拠にしない。
- 20回以上かつ7日以上の連続観測で全期待row `PASS`、他state/cancel/schema error 0、artifact hash一致、p95 12分以下、最大15分以下、unexpected `INFRA_ERROR` 0、missing result 0を満たした後だけH3のrequired昇格をreviewする。それまではnon-requiredで計測し、H0〜H2だけをrequiredとする。
- required昇格観測はG0、GPU runner、model-free runtimeの開発と並行する。7日間を後続実装の開始条件または待機期間にしない。

### Phase 3: GPU runner基盤

- 専用local hostでreview済み完全SHAに対する直列実行とevidence集約を先に実装する。
- H3のrequired昇格観測と並行して、canonical `gfx1030`/`gfx1201`のG0とmodel-free probeを進める。
- runner group、environment、default-branch workflow、ephemeral/JIT登録、host controllerを用意する。
- G0 preflight、process監視、timeout、診断収集、quarantine、reboot/reimageを実装する。
- modelを使わない最小HIP probeで運用を検証する。

実施状況: canonical `gfx1030`/`gfx1201`のG0、read-only health/process observer、完全SHA/treeとexact H3 artifact binding、host lock、G1専用artifact/actual loader検査、timeout/crash/output bound、実行前後health・process確認、fail-closed 2 row aggregateを実装した。commit `f393d688a051d2b73c8773d8a930a711592609bc`（tree `2ccda6e7c0614d585f26babc6b7c68ca51220bbe`）で同一candidateのH0〜H3/G0/G1 aggregateが全てPASSした。

### Phase 4: runtime・kernel test

- 最初のC ABI、allocator、queue/eventと同時にH1/H3/G1を追加する。
- 各semantic opにNumPy oracleとboundary generatorを追加する。
- baseline kernelを正しさの基準とし、optimized kernelを同じcontract suiteへ登録する。

実施状況: semantic opより先に、private evidence ABIのmodel-free allocation/copy/queue-event lifetime/diagnostic kernelをG1へ実装した。canonical 2 GPUで1、3、17、255、256、257 byteをbyte exactに実行し、各caseのallocation/copy/dispatch count、fallbackなし、実行後healthを同一candidateのG1 aggregateで検証済み。semantic opとNumPy oracleは次作業であり、このdiagnostic XOR kernelを数値実装の基準へ昇格しない。

### Phase 5: model slice・end-to-end

- repository内のtiny synthetic fixtureと、固定model cacheから実行時に抽出するlocked real-weight sliceを分ける。
- sliceのsource model lock fingerprint、tensor名、offset/shape、抽出tool repositoryと完全commit SHA、script path/hash、全引数・設定、実行環境、出力path/size/SHA-256を記録する。
- G2を通してからG3を導入する。
- Qwen3.5-4B BF16、single GPU、batch=1、text-onlyに限定し、vision/MTP/multi-GPUを混ぜない。
- model weight/activationはBF16、MVPのKV cacheは連続FP16として別々に検証・記録する。

### Phase 6: compatibility・performance

- G4のtuple evidenceをcompatibility文書とhistoryへ反映する。
- P0を観測として開始し、十分な履歴取得後にP1の回帰閾値を決める。
- llama.cppとの同条件比較を追加する。

## featureの完了条件

新しいop、kernel、runtime機能を完了扱いにするには、該当範囲で次を満たす。

1. contractと異常系が文書化されている。
2. host側のvalidation、error、lifetime testがある。
3. 数値opには独立したtiny oracleと境界caseがある。
4. 対象targetのcompile-only testがある。
5. 利用可能と表記する前に対象実GPU testが通っている。
6. model pathへ影響する場合はmodel sliceが通っている。
7. full model機能として表記する場合は固定model lockのend-to-endが通っている。
8. test時間、seed、tuple、artifactを再現可能に記録している。
9. test未追加、skip、quarantine、未検証範囲を隠していない。

## 未確定事項

- Qwen3.5-4Bの完全commit SHAとmodel lock。
- opごとのaccumulation、丸め、NaN/Inf contractと数値tolerance。
- deterministic RNG injectionを内部test APIへ許可するか。
- performance hard gateを開始するために必要なbaseline回数と閾値。
- test結果・release evidenceのGitHub外長期保存先。

## 公式資料

- [GitHub: Adding self-hosted runners](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners) — 公開repositoryのforkとself-hosted runnerに関する警告。
- [GitHub Actions workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax) — permissions、runner、timeout、matrix、concurrency。
- [GitHub Actions events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows) — `pull_request`、`pull_request_target`、schedule、dispatch。
- [GitHub deployments and environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments) — reviewerとenvironment protection。
- [GitHub self-hosted runners reference](https://docs.github.com/en/actions/reference/runners/self-hosted-runners) — ephemeral runnerとlog forwarding。
- [ROCm GPU isolation techniques](https://rocm.docs.amd.com/en/latest/reference/system-optimization/gpu-isolation.html) — visibility、container、VMによる隔離。
- [ROCm environment variables](https://rocm.docs.amd.com/en/latest/reference/environment-variables/index.html) — GPU visibility設定。
- [AMD SMI CLI](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html) — health、process、reset診断。

[対応する履歴](../../../../../history/2026/08/1-10/ci-test-strategy.md)
