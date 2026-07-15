# P2 active fidelity gate attempt 6: runtime index contract boundary

## 前回の要点

attempt 5は、`HIP_VISIBLE_DEVICES=1`とcapture logical index 0でruntime device guardに到達し、GPU計算前に終了した。serviceは復旧し、raw evidenceをarchive済みである。

## 今回の変更点

commit `895acc948c8b9d20d039011e777f79010b835c57` のGateを独立監査した。Gate SHAは`a955cb01170db46c817638d0771cb70a96e1ce22479f0a25819320c496124b3d`で、physical card2→logical device0、`HIP_VISIBLE_DEVICES=2`、expected `gfx1201`を固定していた。既知のpin、worker nlink1、attempt 5 archive、BASE require_absent、service pre-stateをread-onlyで確認し、同一PTYの`sudo -v`後にGateを一度だけ実行した。

preflightとdevice mapping printは通過したが、captureは`runtime device differs from the pinned active identity`で終了した。GPU compute、24行output、hidden/logits sidecar、metrics、validatorは未生成である。run.logにruntime architectureの詳細はなく、推測によるobserved値は記録していない。

runtime sourceのread-only確認では、`runtime/src/ullm_runtime_api_core.inc`の`ullm_runtime_get_device_info(index)`がindex `0`をCPUとして扱い、index `>0`をHIP index `index - 1`へ変換する。capture sourceは`ullm_runtime_sys::device_info(args.device_index)`を呼び、Gateは`--device-index 0`を渡している。直接出力したROCr観測は`ROCR_VISIBLE_DEVICES=0→gfx1030/V620`、`1→gfx1201/AMD Radeon Graphics`、`2→gfx1030/V620`である。`HIP_VISIBLE_DEVICES=0/1/2`の`rocminfo`はGPU agents `gfx1030,gfx1201,gfx1030`を出力した。これは直接観測として保持し、runtime architectureを推測しない。

serviceは`2026-07-15T09:03:46+09:00`に停止し、`09:04:07`にrestore開始、`09:04:10`にstartup completeとなった。停止窓は21秒。最終状態は`active/running`、MainPID `877038`、NRestarts `0`。`/run/ullm`はmode750・uid/gid1000、lockはinode753708・nlink1・mode600・owner PID877038であり、active/package/worker SHAは維持している。observerは失敗markerなしの1サンプルでGPU useは全card 0%、VRAM usedは21,401,600 / 21,401,600 / 87,384,064 bytesである。

raw evidenceは`benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-calibration-active-v0.1/attempts/active-attempt6-20260715T090346Z/`へ保持し、SHA256SUMSで検証する。Gateは再実行しない。production serviceは変更せず、BASE直下の出力残骸も残さない。

## 次の行動

runtimeのglobal index契約（CPU index 0、HIP indexはglobal index−1）とcaptureの引数契約の是正案を別途レビューする。新しいGPU captureはparentの明示判断と別attemptでのみ実施し、attempt 6の証拠は変更しない。
