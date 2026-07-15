# P2 active fidelity gate attempt 7: capture artifact and post-check boundary

## 前回の要点

attempt 6はruntime global index 0（CPU）を渡していたため、device identity guardでGPU計算前に終了した。runtime APIのCPU global index 0、filtered HIP global index 1という契約を固定し、Gateを再実行せずに証跡をarchive済みである。

## 今回の変更点

commit `1c010e77ab3efc7aaa3d8032c2320a7d2b540fd8`（Gate SHA `776bdc8de72fe7394dfafa03d3ccbca5d7ad16e6461dcc8502eddb31574b8e1b`）を独立監査し、同一PTYの`sudo -v`後にGateを一度だけ実行した。physical card2→HIP visible token1→filtered HIP ordinal0→global device1、expected `gfx1201`を通過し、R9700でGPU計算が完走した。

captureは24行artifactを生成し、Gateのartifact SHA256SUMS検証まで通過した。その後のGate内inline Python verifierが`manifest.row_count`を参照して`KeyError: 'row_count'`となり、Gateはexit90で終了した。生成manifestではrow countは`manifest.cases.row_count`と`manifest.runtime.run.row_count`に存在する。read-only schema auditは、rows 24・unique case IDs 24・unique `(case_id, step)` 24・nonfinite rows 0・model loads 1・one_model_load true・device gfx1201/requested_index1を確認した。sidecarはhidden `393216` bytes、logits `23838720` bytesで、artifact内SHA256SUMSは全OKである。

Gate後のmetrics生成を別途一度試行したが、`aq4_target calibration artifact failed validation: parent sampled oracle binding differs`でmetrics.jsonは生成されず、validatorは実行していない。このpostprocess失敗とstderrをraw archiveに保持した。capture output、markers、run/monitor、metrics stdout/stderrは`attempts/active-attempt7-20260715T091108Z/`へ移し、archive SHA256SUMSを検証した。

serviceは`2026-07-15T09:11:07+09:00`に停止し、`09:11:08`にstop完了、`09:17:46`にrestore開始、`09:17:49`にstartup completeとなった。stop-returnからrestore開始まで398秒（attempt開始から399秒）。最終状態は`active/running`、MainPID `930324`、NRestarts `0`。`/run/ullm`はmode750・uid/gid1000、lockはinode754242・nlink1・mode600・owner PID930324。active/package/worker SHAとworker nlink1は維持している。observerは失敗markerなしの1サンプルで、card2 GPU use 100%・VRAM used `7439605760` bytes、card0/1 GPU use 0%である。

## 次の行動

attempt 7のartifactはmetrics binding不成立として扱い、Gateを再実行しない。inline verifierのschema参照とparent sampled oracleのmanifest bindingは、別の設計修正・レビュー対象として切り分ける。production serviceとraw archiveは変更しない。
