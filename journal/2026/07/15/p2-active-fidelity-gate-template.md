# P2 24-row active fidelity capture gate template

## 前回の要点

P2 source captureの完了待ちで、active AQ4側のGPU/service実行はまだ行っていない。既存 strict attempt3 gate の service、RuntimeDirectory、lock、observer、restore、deadline、identity契約を再利用する必要がある。

## 今回の変更点

main worktreeに専用script/testを追加した。

- `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-calibration-active-v0.1/input/run-active-fidelity-capture-gate.sh`
- `tests/test_qwen35_aq4_active_fidelity_gate_template.py`

基礎commitは `e91a36a0`。続く共有mainの `eec6922f` に、sidecarサイズ・manifest one-load/nonfinite/24-row post-run検証の13行が含まれる（同commitのprofile launcher変更と混在するが、親の内容は保持されている）。

scriptは固定 plan/cases/split/policy/calibration、served/package/worker/guard/device/quantized revision、baseline `ullm-aq4-fidelity-capture` build commitを検証する。source artifact root、source artifact SHA（artifact `SHA256SUMS`のSHA）、source manifest SHA、capture binary SHAは未確定placeholderとして、置換前にpreflight failする。24-row、sidecar上限、disk free、one model load、nonfinite拒否、既存output/log no-overwrite、終了後SHA256SUMS検証、metrics生成・validator前段を含む。通常preflight、locked read-only preflight、mock preflightを分離し、service stopは本番run分岐だけに限定した。

## 次の行動

source artifact完成後に、source root pathと3 SHA placeholderを独立検証して差し込み、plan/casesとactive identityを再確認してfreezeする。その後にのみ本番GPU gateの実行を検討する。
