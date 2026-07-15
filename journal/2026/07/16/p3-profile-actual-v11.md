# P3 profile actual v11

## 前回の要点

- operator-command-v11はcommit `637ca8ed26e8cbb1200656ba4fb6ef1676b8282f`、tree `578f720472e0eef5b5607321e7a21df04fc72cf6`で確定した。manifest raw SHA-256は`4597826e0c876e3b51c756f65c99c2bb43ee395504b7fe9767eb324db1706102`、semantic SHA-256は`623730860c878b7652138bf54b8582677c48a346544244d0ee327b811d4b9387`。
- quiet-window-v16は27/27 clean、span `317.689081904`秒、reset 0でGOだった。直前監査でもmain PID `1212941`、worker PID `1213021`、NRestarts 0、AMD-SMI/KFD ownerはworkerだけ、fresh output 9/9 absentだった。

## 今回の変更点

- double independent GO後、sealed manifestのexact 10 argvを同一PTY、`shell=false`で1回だけ実行した。canonical startは`1784140230351226129`、endは`1784140248388489836`、elapsedは`18037263707`ns、returncodeは1、retryは0。
- full package content hashは1045 files、`7700872459` bytesを1回だけ検証しpassedした。hash後の`pre-stop-snapshot`でGPU owner不一致を検出し、安全に失敗した。maintenance evidenceはowner PID自体を記録していないため、owner identityはnormative evidenceとして断定しない。別のread-only診断では対象外`/tmp/ullm-sq8-main-integration/...` test PID `1558065`を確認した。
- failureはservice停止前で、launcher、capture tool、rocprof、systemctl stop/start、モデルロードはいずれも0回だった。serviceは触られず、main PID `1212941`、worker PID `1213021`、NRestarts 0の同一epochを維持した。execute-evidence-v9、runtime-v9、capture-v9は未生成であり、failure evidenceではabsenceが正規状態。
- 外部test群にはkill/restartを行わず自然終了を待った。複数回のread-only確認でAMD-SMI/KFD ownerがworkerだけ、targeted process 0へ戻ったことを確認した。
- pre-stop no-op recovery対応finalizerをcommit `370ab8cff2fc745d85657260329a80fab21b0acb`、tree `bc967e60213a3cc7080598c2e6de94a9d6df2bb3`、blob `08da9074dc47c6b158f49cf829c463e60a857a81`、raw SHA-256 `42e8be4c15f0eabdc21984d506d7b5fcd9885b637cbb52b9ba206216ac21ad5e`へ固定した。roleは`existing_evidence_recovery_only_not_execution_authority`で、actualやmaintenanceを再実行せず既存streams/evidenceだけをfinalizeした。対象テストは33/33 passed。
- actual audit statusは`failed_immutable_evidence_preserved_restore_passed`、restore classificationは`pre_stop_untouched_same_epoch`。recovery snapshotはsame epoch、hash 3種、formal health、lock identity、AMD-SMI/KFD owner、targeted process 0を確認した。residual process 0、retry 0、secret material false。
- maintenance JSON/SUMS SHA-256は`616a1a7bb9de0109093387856d81e41fa1944eedeaf83a15ad89a1714cd81b66` / `d6dc6cf0df090bcd71d31bf36b02688ac609a1a36c2b6f4071b24cccc9ad2573`。
- operator result JSON/SUMS SHA-256は`9271e2a3e385ac41a1d4ee84b86768863e8deb069628d063b10cdfc4ad610b34` / `907a9c70d3871db8fe079395dd46da75227490c74600a8fea74b3f9c9fba77a1`。stdoutは254 bytes、SHA-256 `003e231a108f42540b504eb5795a4890a619032fe2a9c60edefb1bc4eb719868`、stderrは0 bytes。
- actual audit JSON/SUMS SHA-256は`ca6d1e13939ea90febfa44c0755fdf59add7318a1dd216394126821bd6d12c16` / `5326558991183c08a11519762ea308aad3314f96bc2bd259190aca11ac7536ee`、semantic audit SHA-256は`91710bcb87686aae5712ad47ccf6ce3606dbd1e2ea02a7186be9d8f8668f76a6`。
- 3 evidence rootsは`0555`、全filesは`0444`かつnlink 1で、formal validator、各`SHA256SUMS`、secret pattern scanがpassedした。

## 次の行動

- failure artifact commitを独立QAへ渡し、archiveとGit blobの一致、failure/no-touch境界、finalizer authorityを再確認する。
- actual-v11は既に最大1回を消費したため、再実行しない。次の試行が必要な場合は新しいnamespaceと新しい明示的authorizationを用意する。
