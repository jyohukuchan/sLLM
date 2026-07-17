# P2 baseline/profile measurement lane audit（2026-07-14）

## 前回の要点

- P2ケース展開と閾値テンプレートは、CPUで構造検証できるが、identity・policy・power・same-artifact path oracleの束縛は未生成だった。
- 初回監査では、active workerを停止せずにR9700の実行を始めないことを確認した。

## 今回の変更点

- active状態を読み取り専用で確認した。`ullm-openai.service` と `ullm-aq4-worker` PID 2016033 は稼働中で、R9700（rocm-smi card2、gfx1201、AMD Radeon Graphics）のVRAMを使用している。
- P2 raw-v2 adapterに、policy self-hash/status、identity self-hash/status、identity→policy SHA-256の再検証を追加した。
- adapterが受け取ったpreflightのexact schema、非負値、GPU process snapshotを実行前に検証するようにした。
- full-model driverのargvを固定する`build_driver_command`を追加し、caseの`runtime_device_index`を`--device-index`へ明示的に渡すようにした。`--case-id`も固定し、shell commandを経由しない。
- 専用CPUテストで、policy/identity改ざん、preflight不正、device index付きargvを検証した。

## 検証

- `python3 benchmarks/workloads/validate-aq4-production-opt-p2-manifest.py ...` → `valid=true`（smoke 84 / representative 2245 / full 3885）。
- `python3 -m py_compile tools/prefill_validation/aq4_p2_raw_v2_adapter.py tests/test_aq4_p2_raw_v2_adapter_binding.py` → 成功。
- `python3 -m unittest -v tests.test_aq4_p2_raw_v2_adapter_binding` → 3 tests成功。
- 既存adapter/runner回帰: `test_role_aware_raw_v2_adapter_end_to_end_and_fail_closed_mutations`、`test_runner_rejects_bound_policy_and_package_drift` → 成功。
- GPU実行、active service停止・再起動、live request、R9700 power captureは未実施。

## 残課題と実行条件

- `tools/run-aq4-production-p2.py` は依然としてworker単独argvでrequestを実行しない。P2 full-model driverを実運用するadapterの接続は、raw-v2 adapterの固定argvを使う工程へ集約する必要がある。
- full-model driverのruntimeはHIP resident model前提で、CPU-reference caseが本当にCPU実装として動くかは未確認。unsupported/failedをokへ置換しない。
- driverのVRAM/workspace peakは現raw adapterで0を発行しているため、production measurementへ進む前に実測値を埋める専用captureを追加する必要がある。
- production traceはP1 live-v4が存在するが、P2 identity/policy/path oracleとの同一run bindingはまだない。source/path oracleは入力未生成のため、現時点は明示的にblocked。
- R9700実行前に親エージェントの許可を得て、active workerとの同時使用を避ける。lock (`ullm-r9700-p2-exclusive.lock`)、正のVRAM processの許可リスト、headroom、power/temperature/driver/runtime/hostを同一run rootへ束縛する。

## 次の行動

親エージェントがCPU側adapter/bindingをレビュー後、CPU fixtureまたはCPU実装のunsupported判定を先に確定する。R9700実行計画と排他条件を親へ報告してから、GPU実行を判断する。
