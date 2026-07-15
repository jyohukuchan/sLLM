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

## 検証

- Rust request scope test: 1 passed
- Rust SQ8 counter/fallback test: 1 passed
- AQ4 overlay worker tests: 2 passed
- capture/builder/production trace Python tests: 15 passed
- Python syntax check: passed
- dedicated wrapper lifecycle tests: 6 passed
- strict receipt/generator/builder tests: 6 passed
- combined related Python tests: 44 passed（worktree絶対pathに依存する既存1件は対象外）
- GPU command・service状態変更: 0。wrapper実装確認中にread-onlyの`systemctl cat`を1回実行したが、start/stop/restart/showやGPU commandは実行していない。

## 次の行動

1. Python-only lifecycle/receipt強化をcommitし、新HEADをrelease sourceとして`CARGO_BUILD_JOBS=1`で`ullm-aq4-worker`をrelease rebuildする。
2. strict `prepared_not_executed` receiptを含むcreate-new candidate runtimeへGateを再materializeし、Gate/worker/manifest/receipt SHA-256を固定する。
3. 独立監査が専用wrapper、失敗復旧、actual request telemetry、atomic evidence publicationを承認するまで、`actual_run_allowed=false`を維持する。
