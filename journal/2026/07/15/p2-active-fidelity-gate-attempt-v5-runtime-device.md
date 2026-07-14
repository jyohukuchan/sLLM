# P2 active fidelity gate attempt 5: runtime device identity stop

## 前回の要点

served workerのhardlink別名をatomic detachし、worker pathをSHA一致・regular file・nlink1へ固定した。attempt 4はworker nlink2のためGPU処理へ進まず、Gateは未実行だった。

## 今回の変更点

独立read-only監査で固定plan/cases/split/policy/calibration、source/capture binary、active/package/worker identity、service lock、空のBASE出力を確認した後、同一PTYで`sudo -v`を準備して固定Gateを一度だけ実行した。preflightは通過し、service停止後にcapture binaryを起動したが、runtime device identity guardでGPU計算前に終了した。

- Gate exit code: `90`（cleanup後の終了値）
- run failure: `runtime device differs from the pinned active identity`
- Gate envは`HIP_VISIBLE_DEVICES=1`、captureはlogical device index `0`を要求した
- read-only ROCm SMI観測ではphysical card1がV620、`GFX Version=gfx1030`であり、logical index 0のobserved architectureは`gfx1030`と特定できる
- active manifest、expected device architecture、R9700 card2のpinは`gfx1201`
- capture binaryのコードは`ullm_runtime_sys::device_info(args.device_index)`の`gcn_arch_name`をexpected/pinned architectureと比較し、不一致で上記エラーを返す
- GPU計算、24行active output、hidden/logits sidecar、metrics、validatorは未生成（GPU compute未到達）
- observerは失敗markerなしの1サンプルを保持し、GPU useはcard0/1/2が0%、VRAM usedは21,401,600 / 21,401,600 / 87,384,064 bytes

serviceは`2026-07-15T08:54:14+09:00`に停止し、`08:54:36`にrestore開始、`08:54:39`にstartup completeとなった。停止窓は22秒。最終状態は`active/running`、MainPID `839651`、NRestarts `0`。`/run/ullm`はmode750・uid/gid1000、lockはinode753314・nlink1・mode600・owner PID839651で、active/package/worker SHAは固定値を維持している。BASE直下のoutput/metrics/markersはarchive後に不在である。

raw evidenceは`benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-calibration-active-v0.1/attempts/active-attempt5-20260715T085414Z/`へ移し、`SHA256SUMS`で検証した。Gateの再実行は禁止し、今回のattemptはruntime device binding不一致として終了する。

## 次の行動

追加のGPU captureは、`HIP_VISIBLE_DEVICES`とlogical device indexの組み合わせを是正し、parentが新しい測定判断を明示した場合だけ別attemptとして計画する。今回のGate入力・raw evidence・service復旧結果は変更しない。
