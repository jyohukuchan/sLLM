# AQ4 Phase 3c: guard chain リハーサル v0.1

## 前回の要点

- commit `e3ca6760` で、`runuser` の default PATH が `/opt/rocm/bin` を含まないことに対し、ASIC cross-check と H9 telemetry を `/opt/rocm/bin/amd-smi` の絶対pathへ固定した。
- この変更は service、systemd、active manifest、R9700 lock、07/16停止中P3 harnessに触れず、guardの静的testと既存stage tooling testは通過していた。

## 今回の変更点

- host-only HIP guardを `target/aq4-phase3c-r9700-guard-rehearsal-v0.1/query-hip-device-identity` に buildした。SHA-256 は `e85043b1bc1812a1b0ebcba31fcfa0bff5402be348d713a37f44643d9885175d` である。buildはHIP runtime query用のbinary生成だけであり、kernel launchを行わない。
- `tools/run-aq4-phase3c-r9700-guard.py` を root から3回実行した。各回とも service 稼働中であり、lock、service、systemd、manifestを変更していない。root outputは `benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/guard-chain-rehearsal-v0.1/attempt-{1,2,3}/` に保存した。

| attempt | UTC start | UTC finish | guard | health telemetry |
| --- | --- | --- | --- | --- |
| 1 | 2026-07-17T03:53:49.401452+00:00 | 2026-07-17T03:53:50.819761+00:00 | valid | complete |
| 2 | 2026-07-17T03:54:07.083999+00:00 | 2026-07-17T03:54:08.515226+00:00 | valid | complete |
| 3 | 2026-07-17T03:54:16.317830+00:00 | 2026-07-17T03:54:17.753277+00:00 | valid | complete |

- 3回とも同一の `runuser` command/environment で実行された。PATH はいずれも `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin` で `/opt/rocm/bin` を含まなかったが、絶対pathの AMD-SMI は exit 0 だった。HIP は可視台数1、filtered ordinal 0、`gfx1201`、`0000:47:00.0`、non-empty nameを返し、ASIC cross-check は同一BDF、`gfx1201`、`0x7551`、non-empty nameを返した。HIP/ASIC identity stdout SHA-256も3回で一致した。
- health telemetryの4 command（ECC/clock/power/temperature/perf level、bad pages、driver/IFWI/power limits、firmware）は3回ともJSON parse成功・exit 0だった。attempt 1ではECC/UMC ECCのcorrectable/uncorrectable/deferredが全て0、bad pageなし、throttle=`UNTHROTTLED`、temperature edge/hotspot/mem=`37/38/36°C`、driver=`amdgpu 6.16.13`、IFWI=`00158746`、socket power=`15W`だった。
- 最終window用に、既存evidenceとは独立した `tools/probe-aq4-phase3c-existing-lock.py` と `tools/run-aq4-phase3c-service-window.sh` を追加した。driverは一回の明示confirmを要求し、existing regular lockを no-create probeし、リハーサル済みguardをtrace前後に同じ経路で使う。guard failureまたはtrace failureではretryせず、EXIT trapで `ullm-openai.service` を一回だけstartする。`systemctl restart`、lock作成、V620照会を含まない。
- `bash -n`、`py_compile`、`pytest -q tests/test_aq4_phase3c_r9700_guard.py tests/test_aq4_phase3c_service_window_driver.py tests/test_aq4_phase3c_stage_tooling.py` は6 passed、`git diff --check` は成功した。

## 次の行動

- trace/CPU binary、fixture、fixed trace tooling diff、service pre-stop snapshotを service稼働中に再確認する。
- 新規 evidence leaf を一度だけ作り、上記driverによる Phase 3c service-stop windowを一回だけ実行する。guard、trace、restore、post-restore検証のどこかが失敗した場合は再試行せず、service復旧結果を最優先で記録する。
