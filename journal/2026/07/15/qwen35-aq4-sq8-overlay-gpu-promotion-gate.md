# Qwen3.5 AQ4 SQ8 overlay GPU promotion Gate

## 前回の要点

- SQ8 linear-attention QKV/Z overlayは、48 tensorのartifact、source provenance、CPU oracle、worker admissionまで準備済みだった。
- clean integration candidateは`833a15ce51b2199d9cb35748e29c40069e2c91df`だったが、GPU実行時にSQ8 batch/pairを実際に通ったこと、fallbackとhost stagingが無いこと、service停止・復旧を含むexclusive実行証跡が無かった。
- 既存P1/P2 maintenanceはservice prestate、stable owner-free poll、lock substrate、failure時のunconditional restoreを実装している。一方、P2 one-case/profile diagnostic evidenceはpromotion/fidelity evidenceには使えない。

## 今回の変更点

- `aq4_package_runtime`の既存SQ8 single/batch/pair/triple atomic counterへdispatch fallback counterを追加した。
- `ULLM_SQ8_PROMOTION_EVIDENCE_REQUEST_ID`がrequest IDと完全一致する場合だけ、request開始直前にSQ8 projectionとdiagnostic host-staging telemetryをresetし、terminal request auditへsnapshotするdefault-off経路を追加した。環境変数が無い通常workerは追加reset/snapshotを行わない。
- overlay worker/profileへ`ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL`を追加した。actual Gateではsingle/batch/pair/tripleの全kernel guardを1へ固定し、AQ4 QKV/Z fused gate-betaは無効のままにする。
- `capture-aq4-resident-executor-record.py --sq8-promotion-evidence`はactual requestだけをtelemetry対象にし、実測batch/pair count、unexpected single/triple/fallback、host staging、request ID、token順序とtoken ID列のdomain-separated SHA-256を検査する。token ID自体はevidenceへ保存しない。
- `prepare-qwen35-aq4-sq8-overlay-gpu-promotion.py`は固定commitからbuildされたworkerをsingle-link mode 0555へcopyし、source tree/archive、build command/jobs/toolchain、profile、binding/content/tensor-set、package、worker、served manifestのhash receiptとdeclarative Gateをcreate-newで生成する。
- Gate sequenceはservice prestate、stop、stable owner-free 2回、candidate runtime/lock、pre-hash、overlay load/ready、非対象smoke、telemetry reset後actual request、post-hash、cleanup、default service新epoch/health/lock復旧を必須とする。
- promotion/fidelityは`unclassified`、holdout未使用、policy緩和なしである。Gate自体は`actual_run_allowed=false`で、独立executor/Gate監査が完了するまでGPU/service/systemctl実行を禁止する。
- 独立監査で見つかったlifecycle不足を解消するため、候補専用maintenance wrapperを追加した。default service prestate、停止後のsystemd/worker/AMD/KFD owner-free stable 2回、`/run/ullm/device-1.lock`の同一inode flock、候補capture 1回、source/artifact/package/releaseのpre/post同一性、失敗を含む全経路でのlock解放とdefault service新epoch/health/production lock owner復旧を必須化した。既存P2 production launcherは候補実行に使用しない。
- overlay promotion receiptを2 phase化した。pre-GPU候補は`prepared_not_executed`かつ`actual: pending`で、source commit/tree/archive、worker/profile/served semantic identity、binding/content/tensor-set、完全artifact inventory、packageを束縛する。actual後だけmaintenance stable2とexecutor telemetryを再検証したcreate-new `actual_verified` receiptを生成できる。
- served-model generatorはSQ8 overlayのpromotion profile/receiptをexact schemaで検証し、旧evidence keyの追加、field削除、stale source、worker/profile/manifest、inventory/mode/uid/gid/nlink/bytes、package、actual evidenceの不一致をfail closedにした。
- 最終監査でrequest identityの生成元がcapture側の乱数だった問題を修正した。builderがsource commit/tree/archive、worker、binding/content/tensor-set、packageのcanonical identityをSHA-256へ束縛し、`sq8-promotion-<64 hex>`を1件だけ生成する。Gate、build receipt、prepared receipt、profile、capture argv、worker telemetry環境、executor record、maintenance evidence、actual/failure receiptが同じrequest IDをexact照合する。promotion captureではexplicit `--sq8-promotion-request-id`が必須で、`capture-*`乱数request IDを生成しない。
- prepared receiptは上書きせず、actual成功後だけ別のcreate-new `promotion-actual-receipt.json`を生成する。actual receiptはprepared receiptの絶対path/SHA、maintenance/executorの相対path/SHA、request ID、live worker/profile/overlay inventory/package、stable2 exclusivity、telemetry、manifest/output identityを束縛する。失敗時は別の`promotion-failure-receipt.json`だけを生成する。
- production `generate`/`materialize`は`actual_verified` receiptだけを受理し、GPU前候補は明示的な`generate_prepared_candidate`経路に限定した。wrong request、replay/overwrite、prepared receipt置換、pending receiptのproduction利用を拒否する。
- 独立監査receiptを正式な実行認可入力にした。`--authorize-actual-run`と`--independent-audit-receipt`は常に対で要求し、監査receiptの0444/single-link/non-symlink、source commit/tree/archive、旧Gate/worker/profile/manifest/prepared receipt/SHA256SUMS、request、binding/content/tensor-set/package、`implementation_ready`/`not_executed`をlive identityと照合する。
- 認可候補は監査SHA由来のcreate-new固定pathだけへ生成し、Gateを`authorized_pending_execution`、`actual_run_allowed=true`、`max_attempts=1`にする。Gate、prepared receipt、profile、served manifestへ同一の監査絶対path/lowercase SHA-256を伝播し、片側flag、identity mismatch、writable/symlink audit、出力replayを拒否する。
- gatewayは`promotion.authorization_audit`をtyped optional identityとして読み、旧manifestのabsent/null互換を維持しながら、object時はexact path/SHA、canonical absolute regular non-symlink file、lowercase SHA-256、live hashを検証する。SQ8 profileはauthorization auditのreceipt mappingを必須にし、認可候補ではnullを許さない実体bindingを生成する。
- maintenance wrapperは未認可候補のactual executeをservice参照前に拒否する。認可候補ではGate/receipt/manifestの監査identity一致、監査ファイルの0444/single-link/non-symlink/live SHA、`max_attempts=1`を再検査する。
- 最初のauthorized preflightで、host namespaceから`172.20.0.1:8000/readyz`へ到達できず`healthy=false`になった。service、worker、production lock、AMD/KFD ownerは正常で、actual commandは発行していない。
- host `urllib` readinessを廃止し、既存の`open-webui` bridge namespaceだけからreadyを観測するfail-closed経路へ置き換えた。builderがcontainer name/full ID/image ID/config image、network name/full ID/driver/bridge interface、固定endpoint/status/body/body SHA/timeoutをread-onlyで取得し、request ID、Gate、profile、prepared receipt、served manifestへ同一objectを束縛する。
- wrapperはGate/receipt/manifestのreadiness exact一致後、Docker container/network inspectとlive bridge interfaceを照合し、full container IDを指定したbounded `docker exec curl`だけで`/readyz`を検査する。HTTP 200と本文`{"status":"ready"}`のbyte完全一致を要求し、host fallback、iptables/nftables変更、container名だけの実行を許可しない。prestateとpost-restoreは同じpredicateを使う。
- bridge readiness修正後の最初のone-shotは、service停止でsystemd `RuntimeDirectory=/run/ullm`自体が削除された後、非root wrapperが`/run/ullm/device-1.lock`を作成できず、GPU request前に失敗した。`actual_run_count=0`、executor recordなしで、failure receipt SHA-256は`c37c45c5a0975107cfa8033757f4ed9d45ab0ced8377daf5cb57f2e6effbeb30`。このauthorizationは消費済みで再利用しない。
- root専用の固定path lock helperを追加した。service stopped/owner-free後だけ、`sudo -n`のexact argvで`/run/ullm`を0750 uid/gid 1000、`device-1.lock`をcreate-new 0600 uid/gid 1000 single-linkとして作る。wrapperは同userでopen/flockし、helperが返したdevice/inodeとの一致をcapture終了まで検証する。cleanupはservice stopped中に同inode lockだけを削除し、wrapper-created directoryがemptyの場合だけrmdirしてからserviceをstartする。
- restoreは最大120秒のmonotonic deadlineで、一時的なsystemd/cgroup topology例外を含めてretryする。active/running、新main PID、`NRestarts=0`、新cgroup worker、production lock owner、AMD/KFD owner、Gate-bound bridge healthを順に要求し、attempt count、elapsed、last failure、全observationをmaintenance evidenceへ残す。
- 新request/Gate/build receiptは、消費済みfailure receiptのimmutable path/SHA、prior request ID、`consumed_failed_not_reusable` dispositionをauthorization lineageとして束縛する。

## 検証

- Rust request scope test: 1 passed
- Rust SQ8 counter/fallback test: 1 passed
- AQ4 overlay worker tests: 2 passed
- capture/builder/production trace Python tests: 15 passed
- Python syntax check: passed
- dedicated wrapper lifecycle tests: 6 passed
- strict receipt/generator/builder tests: 6 passed
- combined related Python tests: 44 passed（worktree絶対pathに依存する既存1件は対象外）
- final request/receipt auditを含むcombined related Python tests: 46 passed（同じ既存1件は対象外）
- GPU command・service状態変更: 0。wrapper実装確認中にread-onlyの`systemctl cat`を1回実行したが、start/stop/restart/showやGPU commandは実行していない。
- authorization builder/receipt/wrapper tests: 18 passed
- openai-gateway full tests: 237 passed
- 独立監査receipt実物照合: passed（SHA-256 `db71e280e6605118883f2de80ed308df85dc03a1b9b8b79f947dbc106cfa5146`）
- 統合Python tests: 57 passed。既存deployment profileが元worktreeの絶対pathを持つため、別worktreeでは既存1件だけ失敗する。
- bridge readiness builder/receipt/wrapper関連: 40 passed
- gateway full regression: 247 passed

## 次の行動

1. bridge readiness identity統合を通常commitとして固定し、`CARGO_BUILD_JOBS=1`でworkerをrelease rebuildする。
2. fresh unauthorized runtimeをcreate-new materializeし、Gate/worker/manifest/receipt/SHA256SUMSとmodeを固定する。旧authorized runtimeはstaleとして使用しない。
3. 新runtimeを独立監査し、新audit receiptから新しいauthorized runtimeを生成するまでGPU/service実行を禁止する。
