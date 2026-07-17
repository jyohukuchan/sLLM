# P2 capability-only stage gate

## 前回の要点

- smoke stageはCPU-referenceだけで、HIP-only served workerとの組み合わせはruntimeを触らず`unsupported`にする必要がある。
- 既存validatorはstatus集合に`unsupported`を含むものの、通常の2 warmup + 10 measured、timing、performance、state、calibrationをstatus非依存で適用していた。

## 今回の変更点

- `validate-aq4-production-p2-evidence.py`に`ullm.aq4_p2_capability_record.v1`を追加し、CPU/HIP mismatchのimmutable capability rowを検証する経路を追加した。
- capability rowはserved worker manifest/binary hash、reason code、`model_loads=0`、`gpu_processes=0`、raw immutable statusを必須化し、通常のmeasurement/timing/performance/correctness/path/trace判定から除外する。ただしscheduled matrixのresult_count/missing再計算には残す。
- `reference_source_oracle`ではmanifest/payload/detached validator reportの3リンクを同一hash束として要求し、`sq8_0_cross_format`では`sq-fp8-artifact-v0.2` identity/evidenceとsource-correctフラグを要求する。
- 変異テストでmodel load、reason code、source validator欠落、SQ8 schema改変を拒否する。

## 次の行動

- runnerまたはresult builderがcapability rowをこのschemaとlinksで発行する実装を別途束ね、smoke stage transitionにstatus分岐を接続する。
- 現行のsource-oracle-v2 manifestは旧形式でvalidator reportを含まないため、3リンク契約へ再束縛するまでsource control gateはblockedとする。

## 検証

- `pytest -q tests/test_validate_aq4_production_p2_capability.py`（3 passed）
- `python3 -m py_compile tools/validate-aq4-production-p2-evidence.py tests/test_validate_aq4_production_p2_capability.py`
- 既存`tests/test_aq4_production_p2_evidence.py`は既存 calibration fixture binding差分で失敗（今回のvalidator経路外、親へ報告）。

