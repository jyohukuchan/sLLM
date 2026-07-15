# P3 profile quiet window v3

## 前回の要点

attempt4後のproduction serviceはMainPID `2634680`、worker PID `2635236`、`NRestarts=0`、start epoch `Wed 2026-07-15 16:02:46 JST`、lock device/inode `26:772895`でactive/runningへ復旧していた。P3 v3 actualの前に、この現epochと外部操作のquiet-windowを改めて固定する必要があった。

## 今回の変更点

read-only collectorでformal container healthを開始・終了に実行し、その間を5秒間隔25 samplesで監視した。各sampleはsystemd MainPID/SubState/NRestarts/start epoch、worker PID、lock identity/holder、AMD-SMI/KFD owner、外部systemctl/maintenance/profile capture/rocprof/GPU probe process、全`pts/N`端末を個別の`ps -t pts/N`で記録した。

production runtimeは全samplesで不変だった。serviceはMainPID `2634680`、worker `2635236`、`NRestarts=0`、lock inode `772895`/holder `2634680`、AMD/KFD owner `2635236`を維持した。formal healthは開始・終了ともcontainer identityが一致し、gateway healthz/readyz/modelsとOpenWebUI healthは全てHTTP 200、process countはDocker 9、docker exec 6、container curl 6だった。全pts process setも不変で、対象の外部process観測は0件だった。

ただしwindow中にGit HEAD/treeが `165b81aa...` / `77e7d3b4...` から `d215d4b3...` / `db97041c...` へ変更された。`702f67b9`、`a50f9973`、`d215d4b3`の3 commitが同時刻に追加されている。またcollector全体のmonotonic elapsedは約127.48秒だったが、25 samplesのfirst-to-last timestampは `119.970106854` 秒で120秒契約を満たさなかった。fail-closed判定は `NO-GO`、violationsは2件である。

機械可読証拠は `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-quiet-window-v3/quiet-window.json` に0444で保存した。SHA-256は `19ae43dd6c791c7fe0754e35cdbf7a43f396686b6fbebbbf106f68439386e8ef`、`SHA256SUMS` 自体のSHA-256は `8730b7364d2e95374fb50b95cb2ee5705d8715aeebb5163ee125171262a4518c`である。secret materialは記録していない。

## 次の行動

このNO-GO evidenceをGOへ読み替えず、P3 v3 actual・GPU command・service操作を実行しない。repo commit activityが停止した後に、first-to-lastが確実に120秒を超える新しいfresh quiet-windowを別versionで取得する必要がある。
