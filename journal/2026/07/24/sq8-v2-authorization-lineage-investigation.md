# SQ8_0 v2 authorization/lineage investigation

## 前回の要点

独立フォーマット`SQ8_0`の本番昇格では、2026-07-12のcomplete campaignが
履歴worker `145a5351...b950`とlegacy起動方式に固定され、現行candidate
workerとmanifest起動方式の証跡にならないことが判明していた。現行profile
は`ullm.worker.v1`かつreasoningなしで、生成結果も
`ullm.served_model.v1`だった。ユーザーはv1例外ではなくv2で進めると決定
した。

## 今回の変更点

- `/etc/ullm/served-models/active.json`、AQ4 promotion runbook、profile、
  generator、Python/Rust loader、activation/bundle toolingを読み、
  AQ4_0本番の実際のv2経路を特定した。
- served-model v2の共通部分は、厳密な`ullm.served_model.v2` manifest、
  `ullm.worker.v2`、reasoning contract、promotion receipt hash、
  generic-reasoning release bundle、atomic activation/rollbackである。
  profile wrapper自体はAQ4 v2でも`ullm.served_model.profile.v1`のままで
  あり、SQ8向け`profile.v2`を新設する必要はない。
- 現在のAQ4 bytesは正式なcomplete bundle経路ではなく、同一model ID
  限定のdiffering-worker v2 bootstrapで稼働していることを確認した。
  そのsidecarは監査用でruntimeには読まれず、任意の新規pathを使えば再実行
  できるためscheme-wideなone-shot認可でもない。現行実装はmodel ID差を
  先に拒否するので、AQ4から独立SQ8へのcandidate-active切替にはそのまま
  使えない。
- `0cd6b9a0`から`6ad51ac5`までの「SQ8 authorization lineage v2」は、
  current mainと非連結のside historyにあるQwen3.5 AQ4_0の48個QKV/Z
  tensor向けSQ8 overlay専用実装だと確定した。AQ4 worker、overlay
  promotion/audit schema、固定履歴、request ID、48-tensor topologyを
  hard-codeしており、独立SQ8_0へ再利用してはならない。
- 現行generic bundle v1のidentity/rollback/独立validator設計とatomic
  activationは再利用できるが、exact 6-slot envelopeにはSQ8 full campaign
  の格納先がない。pre-receipt promotion evidenceへ事後campaignを結び付ける
  こともできないため、SQ8 campaign 3 referenceを追加したbundle v2と
  activation側の明示schema dispatchが必要である。現行bundle内promotion
  validatorはAQ4 schema専用で、candidate receiptとの直接cross-checkと
  browser evidence内のmanifest/worker identityも欠けている。
- SQ8 worker protocol parserはv2 reasoningを受理できるが、現行SQ8 serving
  runtimeは`reasoning_usage: None`を返すため、reasoning requestのrelease
  accountingを通せない。worker再ビルドとは別にreasoning state/accounting
  実装とCPU testが必要である。またv2 profileでもdecoderがv1 commandを
  明示compatibility modeなしに受理できるため、loaded schemaとの一致を
  強制する必要がある。
- Qwen3-14B-FP8 tokenizerのthinking tokenをread-only確認し、
  `<think>`=`151667`、`</think>`=`151668`を得た。これを用いた
  `qwen3-thinking-v1` contract案は、budget/forced-close/history/answer
  reservationを含むruntime test後に確定する。
- runbook
  `docs/plans/sq8-recovery-plan-v0.2-promotion-runbook-v0.1.md`へ、v1/v2
  比較、旧overlay lineageの切り分け、SQ8 serving promotion
  evidence/receipt構造、対応コード箇所、他served-modelとの互換条件、
  bundle v2、事前発行authorizationのatomic claim、campaign全体を包む
  cross-model temporary window、locked rollback、実装順、admission
  checklistを追記した。
- この調査時点では、実装コード、service、GPU、systemd、V620、active
  manifest、artifact、candidate、worker binaryには変更を加えていない。

## 次の行動

1. 別作業のworker再構築結果をreproducibility baselineとして受け取る。
   reasoning/auth変更後、同一final release commitからworker/build receiptを
   改めて作り、これだけを昇格identityにする。
2. Qwen3 reasoning contractとexact worker-schema enforcementをCPU上で
   実装・検証し、人間がdialect/budget semanticsとversioned v2 specsを
   承認する。
3. SQ8 serving promotion evidence/receipt tooling、no-clobber publication、
   generatorの厳密なcurrent-main AQ4/SQ8 dispatchを実装する。
4. generic reasoning/browser gateのprocess/identityをvalidated manifest
   由来にし、各stageで実`active.json` bytesをcandidateと比較する。
   SQ8 full campaign v2はcandidate copyを含めてend-to-endで束縛する。
5. generic bundle v1/AQ4を不変で保持し、SQ8 full campaign用3 slot、
   candidate receipt/browser identity cross-checkを持つbundle v2を追加する。
   独立再計算、AQ4回帰、mixed-schema拒否を確認する。
6. 人間がexact identity/run/output/expiry/max_attempts=1を持つauthorization
   を事前発行し、固定registryでatomic claimする仕組みを作る。campaign全体
   をactivation lockと`finally` restoreで包み、AQ4 reverse reconciliation
   とimmutable outcomeを私有copy上の全failure境界で検証する。
7. 認証修正を含む単一clean commitからfinal worker/candidateを凍結し、
   parent-onlyでfresh SQ8 full campaignとreasoning/browser campaignを
   実行する。AQ4 exact bytesへの復旧成功後にcomplete bundle v2を組み、
   最終昇格だけを`--release-bundle`で行う。

## 追記: v2実装完了後の状態

上記は調査時点の記録である。その後、独立Qwen3-14B-FP8 `SQ8_0`向けに
次を実装した。旧Qwen3.5 `AQ4_0`の48 QKV/Z tensor向けSQ8 overlay系統は
取り込んでいない。

- Qwen3 reasoning/accounting、worker-v2 discriminatorのexact一致、ratified
  manifest/worker/session/acceptance/release specs。
- strict SQ8 serving-promotion evidence/receipt、AQ4/SQ8 exact dispatch、
  各stageの実`active.json` bytesとcandidate copy/claim/run/outputの束縛。
- generic reasoning evidence v2、browser evidence v5、SQ8 full campaign
  identity v2、およびAQ4 six-slot bundle v1と共存するSQ8 nine-slot
  `ullm.generic_reasoning_release_bundle.v2`。browser v5はPlaywright
  runner imageとOpenWebUI server imageを分離し、browser実行直前・直後の
  fixed server image ID/config/name/running stateを記録する。
- hash由来固定registryのatomic one-shot claim、source-bound fixed plan、
  exact-six campaignと単一expiry deadline、pinned dirfdと
  `renameat2(RENAME_EXCHANGE)`、subreaperを用いるlocked transaction。
- claim SHA-256由来labelを全transient Docker containerへ強制する
  source-bound wrapper、daemon遅延createを含むquiescence/一括cleanup、
  AQ4復旧前と最終live proof前のzero-container証明。短縮/clustered
  `-l`によるlabel上書き、producer単位のwrapper bypass、旧
  `sudo`/`nsenter` gateway probeを拒否・除去した。
- fresh SQ8 3 campaign後のexact AQ4復旧/reconciliation、続くfresh AQ4
  reasoning/browser/bundle-v1 3 campaign、immutable outcome。旧AQ4
  producerを含むevidence producerだけを固定service identityへdropし、
  service-owned random private stagingへ隔離する。全descendant reap後に
  rootがdescriptor-walk、adopt、freeze、validateし、authorized final
  pathへno-replace発行する。AQ4/SQ8の異なるsource lineageは明示的に
  分離した。
- authorizationのOpenWebUI imageとfixed planを同一digestに固定し、
  compose後のrunning container ID/image ID/config/name/running
  state/PID/start timestampも各browser evidenceの直前・直後にread-only
  検証する。
- root実行toolの一時差替えを防ぐため、SQ8/AQ4双方のcampaign sourceを
  protected ancestry配下のroot-owned standalone cloneに限定し、worktreeと
  in-tree `.git`を再帰fingerprintする。linked worktree、symlink、ACL、
  hardlink、object alternate、group/world writeを拒否し、runner自身も
  sealed SQ8 cloneから起動する。recoveryは実行するsealed SQ8 code、
  AQ4 backup runtime、shared unit/environment/credential/rollback operation
  だけを要求し、旧AQ4 sourceやdisplaced SQ8 runtimeを要求しない。
- manifestが参照するworker、promotion pair、tokenizer、product/package
  manifestと全payloadもroot-owned protected ancestry下のruntime closure
  としてsealし、transaction/recovery/final activationの全command境界で
  repinする。現行AQ4 hash `5d015a...`のclosureはUID 1000のworker/
  receipt/evidence、0664のtokenizer 4 files、package 1,044 files
  （約7.70 GB）をuser-owned tree下に持つため、claim前preflightで意図的に
  rejectされる。
- signal defer、expiry後にも使えるlocked recovery。復旧成功にはAQ4
  bytesに加え、service/boot epoch、gateway/worker
  PID・PPID・starttime・executable hash、Gateway/OpenWebUI
  health/modelsのstructured live proofを要求する。
- candidate readiness後に最低900秒、authorization/claim、source、
  candidate/runtime、実active bytes、service/gateway/worker epochを監視し、
  drift無しを確認してからSQ8 fullを開始する。SQ8 fullの上限は21,600秒で、
  authorization残時間が常にさらにcapする。
- source-owned Pythonは`/usr/bin/python3.12 -I -S -B`、root-owned
  sibling importが必要なROCm vendor scriptは
  `/usr/bin/python3.12 -E -S -B`へ固定した。OS/Python/ROCm/ELF、
  Git/Docker/systemd内部とcontainer runtimeは明示TCBである。
- browser JWTをroot-owned parent
  `/run/ullm-campaign-secrets`（`uid=0,gid=1000,mode=0750`）配下の
  `openwebui-session.jwt`（`uid=0,gid=1000,mode=0640`）へ固定し、
  same-UIDによるpathname差替えを拒否する。
- `succeeded_restored` exact-six outcome、fresh AQ4 bundle v1、complete
  SQ8 bundle v2、exact AQ4 rollbackを前提にしたfinal activation plan v2、
  default read-only preflight、exact plan SHA/literal confirmation、および
  locked rollback。final activation runner自身もmodule-derived
  root-owned standalone source sealとしてplanへcommit/treeを固定する。

実装はCPU/private-copy/mock testまでで、本番`active.json`、
`ullm-openai.service`のlifecycle、systemd設定、GPU、V620は変更して
いない。調査中に`systemctl show`相当のread-only metadata queryを一度
実施したが、start/stop/restartやunit/environment変更は行っていない。
production authorization/claim、fresh campaign、final
activation/rollbackも未実行で、real OpenWebUI browser-session JWTは
未用意である。

この追記時点では、全実装を含む単一clean commitからの最終worker
rebuild、build receipt、SHA-256、最終artifact pathは未確定である。既存の
`uLLM-sq8-manifest-candidate-release-ee62d04e`はbaselineのまま保持し、
最終identityとして再利用しない。

次の人間作業順は、(1) 別authorization/lock/rollback/live proofを持つ
AQ4-to-AQ4 runtime-hardening promotionを実施し、root-owned closure向けの
fresh promotion pairと新しいAQ4 manifestを作る（GPU/service windowが
必要）、(2) clean detached commitからfinal SQ8 workerを
build/freeze、(3) SQ8 promotion pair/profile/candidateを別pathへ
no-clobber発行してcomplete closureをroot-stage、(4) exact commitの
SQ8/AQ4 standalone cloneを別のroot-owned pathへ封印してsealed SQ8側の
runnerからpreflight、(5) JWTを用意してhardened AQ4を`before`にした
exact-six v2 authorizationを事前発行、(6) 承認済みmaintenance windowで
serviceを停止しfixed inactive setをread-only確認してから初めてclaimする、
(7) locked transaction内でfresh SQ8 3 campaign、exact AQ4復旧、fresh
AQ4 3 campaign（bundle v1を含む）を実行し`succeeded_restored`を確認、
(8) SQ8 complete bundle v2を独立検証、(9) outcome由来AQ4 bundle v1と
SQ8 bundle v2を再検証するfinal plan/read-only preflightをreview、
(10) Claude+ユーザー明示承認後だけactivation実行、である。

## 追記: final admission / crash-recovery audit

上の実装完了追記の後、production admission と final activation の
failure boundaryをもう一度監査し、次を追加でhardeningした。ここでいう
SQ8は独立`SQ8_0`だけであり、旧AQ4 partial-FP8 overlayとは無関係である。

- 通常のserved-model generatorは、worker-v2のAQ4/SQ8について最終
  format selectorを必須にしてfail closedとした。歴史互換は既存の
  selector無しSQ8 worker-v1だけに限定した。promotion前の一時manifestは
  AQ4/SQ8それぞれの専用non-CLI APIと厳密なephemeral receipt schemaへ
  分離し、final candidate receiptとして流用できない。
- v2 evidence producerはtransaction-private stagingを常に必須とし、
  新規campaignの各admissionで期限・対象・入力を再検証する
  `load_live_claim`を使う。完了済みbundle/outcome/recoveryのauditと、
  expiry後にもAQ4を復旧させるrestore/recovery repinは、authorization
  hashから導出した固定claim registry/UIDに対するarchival `load_claim`
  を使えるが、それだけで新規campaignを認可しない。任意claim pathnameは
  信頼しない。
- 実`active.json`の観測対象は
  `/etc/ullm/served-models/active.json`に固定した。transactionは
  preflight、candidate switch直前、各repinでfinal SQ8 promotion
  receipt/evidence/manifest bindingを完全再検証し、promotion前の
  ephemeral scaffoldをcandidateとして拒否する。
- root control wrapperは、campaign-local importとargument parsingより前に
  exact root `/usr/bin/python3.12 -I -S -B` invocation、ambient interpreter
  configuration、protected ancestry、single-link source filesを検査し、
  clean sealed sourceからだけ実行する。
- claim、outcome、recovery、AQ4 backup、bundle、activation receiptの
  immutable publicationは、temporary hardlinkを作らず
  `renameat2(RENAME_NOREPLACE)`、file/parent `fsync`、format固有の
  post-commit再検証を使う。policy-owned authorization/final receiptは
  ownerも検証し、bundleはexact bytes/mode/`nlink == 1`を検証する。
  rename後のfaultを未commitと誤認して同じone-shot出力を再利用しない
  fault testも追加した。
- final activationは`ullm.served_model.final_activation_plan.v3`、
  operations v2、outcome v2へ更新した。credential sealをswap前に完了し、
  immutable `final_activation_intent.v1`をdurable publicationしてからだけ
  candidate bytesへ交換する。成功receipt publicationがcommit boundaryで、
  その後にAQ4へ戻し得るfallible source checkは存在しない。
- intent後のSIGKILL/power-loss、`failed_restore`、および
  `rollback_incomplete`には、同じlockとexact AQ4 bytesを使う明示確認付き
  recovery routeを追加した。失敗したrecovery attemptはplan-bound baseと
  random 256-bit attempt IDから導出した別のimmutable audit/proofへ残し、
  成功receipt pathnameを消費しないため再試行できる。

統合後のCPU/private/mock回帰では、dispatch/promotion 137件、
bundle/validator 82件、authorization/producer 203件（22 subtests）、
locked transaction 125件、final activation/recovery 100件、
SQ8 full production contract 25件（16 subtests）、追加AQ4/served-model
互換77件、gateway 241件が通過した。Rust側はこのfinal auditで変更して
おらず、直前の全workspace回帰が通過済みである。全`tests/`一括実行は、
現行AQ4 bootstrap identityと過去固定fixtureの既知の不一致を含むため、
「全件green」の根拠には使っていない。

この監査中もproduction GPU、service lifecycle、systemd設定、
`active.json`、JWT、campaign、activationは変更・実行していない。
既報のread-only systemd metadata queryに加え、transaction unit testの
隔離修正前に一度だけtest固有labelでDocker inventoryをread-only照会した。
該当containerは0件で、removeその他のmutationは無かった。その後のtestは
daemonへ接続しない境界へ修正済みである。

このjournalを含む最終clean commitを固定した直後、そのcommitのdetached
clean cloneから新しい別pathへworker releaseをbuildし、release内のbuild
receipt、seal、worker SHA-256を外部artifactとして固定する。build後には
source commitを変える追記commitを作らず、exact path/hashはoperator報告と
release receiptを正とする。既存
`uLLM-sq8-manifest-candidate-release-ee62d04e`は引き続きread-only baseline
であり、削除・上書き・最終identityへの流用をしない。

このbuild順序は、上の暫定「次の人間作業順」にあったAQ4 hardeningとSQ8
buildの順序を置き換える。今回SQ8 final releaseを先に固定し、以後の人間
作業はAQ4-to-AQ4 hardeningから開始して、固定済みSQ8 releaseをその後の
protected runtime stagingへ使う。
