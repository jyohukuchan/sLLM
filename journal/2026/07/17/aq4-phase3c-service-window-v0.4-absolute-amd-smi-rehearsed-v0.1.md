# AQ4 Phase 3c: absolute amd-smi リハーサル後 service window v0.4

## 前回の要点

- commit `e3ca6760` で、`runuser` の PATH に依存せず `/opt/rocm/bin/amd-smi` を使うR9700 guardへ変更した。
- commit `d0f99eec` と `guard-chain-rehearsal-v0.1` で、service稼働中の同一 `runuser` 経路によるHIP identity、ASIC cross-check、H9 telemetryを3回連続で成功させた。対象は常に `HIP_VISIBLE_DEVICES=1` → filtered ordinal 0 → `gfx1201` / `0000:47:00.0` / `0x7551` だけであり、V620を列挙・照会する command は実行していない。
- 07/16に停止したP3 harnessのlock、root、artifact、環境変数、`rocprof`には触れない。

## 今回の変更点

### 最優先: service 復旧結果

- `ullm-openai.service` の stop は `2026-07-17T04:03:19+00:00` に一回だけ成功した。start は `2026-07-17T04:03:22+00:00` に一回だけ呼ばれ、`2026-07-17T04:03:23+00:00` に成功した。restart、追加stop/start、trace retryは実行していない。
- post-restore の service は `active/running`、MainPID=`1128520`、worker PID=`1128926`、`NRestarts=0`、manifest SHA-256=`feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44` で停止前と一致した。healthz は `{"status":"ok"}`、readyz は `2026-07-17T04:03:39+00:00` に `{"status":"ready"}` へ到達した。
- worker は `/dev/kfd` FDを保持し、R9700 BDF `0000:47:00.0` だけを対象にした post-restore AMD-SMI process query は worker PID=`1128926` を owner、VRAM=`7351832576 B` と記録した。lock は mode `0600`、uid/gid `1000:1000`、device/inode=`26:889811` のまま、MainPID=`1128520` がkernel FLOCK WRITE holderである。
- stop開始からstart command成功まで約4秒、readyz成功まで約20秒、完全なpost-restore snapshotの確認完了まで約61秒だった。service復旧は成功である。

### stop後の lock と R9700 guard

- stop後 `2026-07-17T04:03:19.835135+00:00` の no-create lock probe は、existing regular file、mode `0600`、device/inode=`26:889811` を確認し、`O_RDWR|O_NOFOLLOW|O_CLOEXEC` と `LOCK_EX|LOCK_NB` で取得・解放に成功した。`create_flag_used=false` であり、RuntimeDirectoryPreserve=yes の効果はこのwindowでも再確認できた。
- trace前の `guard-before` とtrace後の `guard-after` はともに valid、health telemetry completeだった。HIPは可視台数1、filtered ordinal 0、`gfx1201`、`0000:47:00.0`、non-empty nameを返し、ASICは同一BDF、`gfx1201`、PCI device ID `0x7551`、non-empty nameを返した。すべて `/opt/rocm/bin/amd-smi` の絶対pathであり、runuser PATHは依然ROCmを含まない。

### trace 実行結果

- trace invocation は `2026-07-17T04:03:21+00:00` に一回だけ開始し、同秒に exit code `1` で終了した。GPU kernel trace、comparison、retryは実行していない。
- `gpu-trace.stderr` のfailureは `Qwen3.5 AQ4 differential trace failed: trace binary must be a regular file` である。実体は `target/release/ullm-aq4-differential-trace` の regular file（mode `0700`）だが、nlink=`2`だった。trace binary側のidentity contractは regular fileに加えて nlink=`1` を要求するため、`env::current_exe()` のSHA-256取得前にfail-closedした。これによりHIP runtime device buffer、stream、kernel launchには到達していない。
- このnlink preflightはservice停止前に検証されていなかった。single-use契約に従い、このwindow内でbinaryを作り直す、nlinkを変更する、traceを再実行することはしていない。

### H9 telemetry

| phase | power | throttle | gfx/mem clock | edge/hotspot/mem | ECC / UMC ECC | bad pages | perf level |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| before | 13 W | UNTHROTTLED | 1103 / 96 MHz | 38 / 38 / 36 °C | all correctable/uncorrectable/deferred = 0 | none | AUTO |
| after | 13 W | THROTTLED | 96 / 96 MHz | 37 / 38 / 36 °C | all correctable/uncorrectable/deferred = 0 | none | AUTO |

- driver=`amdgpu 6.16.13`、IFWI=`SAPPHIRE RADEON AI 32GB` version `00158746`、firmware（CP_PFP=2950、CP_ME=2880、CP_MEC1=3200、RLC=12484000、SDMA0/1=7966358、VCN=09.10.B0.01、PSP_SOSDRV=00.3A.10.14、TA_RAS=1B.3A.00.01、PM=00.104.75.00ほか）を保存した。
- ECC・bad page・温度には異常を示す値はない。一方、trace後（ただしGPU kernel launch前）のidle clock時に`THROTTLED`表示があった。この一観測だけではH9を支持も否定もできず、実負荷traceが未実行であるためH9は判定不能である。

### 段階比較と仮説判定

| stage | relative L2 | cosine | max abs | threshold 判定 |
| --- | ---: | ---: | ---: | --- |
| qkv_dequant_row_scale | 未測定 | 未測定 | 未測定 | trace未起動 |
| z_dequant_row_scale | 未測定 | 未測定 | 未測定 | trace未起動 |
| recurrent_gate | 未測定 | 未測定 | 未測定 | trace未起動 |
| recurrent_beta | 未測定 | 未測定 | 未測定 | trace未起動 |
| recurrent_state_after | 未測定 | 未測定 | 未測定 | trace未起動 |
| recurrent_output | 未測定 | 未測定 | 未測定 | trace未起動 |
| attention_residual | 未測定 | 未測定 | 未測定 | trace未起動 |
| post_norm | 未測定 | 未測定 | 未測定 | trace未起動 |
| mlp_activation | 未測定 | 未測定 | 未測定 | trace未起動 |
| layer_output | 未測定 | 未測定 | 未測定 | trace未起動 |

- 最初に有意な乖離を示したstageは特定不能である。H5（GPU kernel固有バグ）は判定不能であり、支持・否定のどちらにも更新しない。H9も上記のとおり判定不能である。

### evidence と実行した検証

- evidence root: `benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.4-absolute-amd-smi-rehearsed/`
- service稼働中preflight: fixed trace tooling commit `5a0fb4c50476d5153ced22bd6847c2729bfdb975` の対象diff/indexがclean、`cargo build --release -p ullm-engine --bin ullm-aq4-differential-trace --bin ullm-aq4-layer0-family-isolation`成功、fixture verifier=`{"cases":3,"status":"valid"}`、CPU final-context frames=30。
- stop window driver: `sudo bash tools/run-aq4-phase3c-service-window.sh OUT HIP_GUARD_BIN --confirm-single-window` を一回実行した。driverはlock probe、同一R9700 guard、trace、post-trace health guard、単回restoreを記録した。
- post-restore: `systemctl show`、Docker bridge namespace経由のhealthz/readyz、service cgroupのworker/KFD FD、`/opt/rocm/bin/amd-smi process --gpu 0000:47:00.0 --general --json`、targeted lock holderをread-onlyで確認した。
- このwindow後にsource、service/systemd/manifest、P3 harnessを変更していない。

## 次の行動

- この単回windowのtraceはnlink identity preflightで無効となったため、同一windowでも新規windowでも再試行しない。
- 追加のGPU window、trace binary identity preflightの改善、H5/H9の再判定、Phase 4以降のfix実装には、別途明示的な判断と承認が必要である。
- 現在の最優先状態はservice正常復旧であり、これを維持したまま今回のevidenceを報告する。
