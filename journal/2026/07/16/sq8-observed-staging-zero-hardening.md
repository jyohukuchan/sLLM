# SQ8 observed telemetry staging zero hardening

- 前回の要点: timeout evidence の trusted component、memfd、fd 検証、safe-int telemetry は監査済みだったが、runner の failure observed telemetry だけが staging の非ゼロ値を受け入れていた。
- 今回の変更点: runner の observed `diagnostic_host_staging` で、4カウンター（read/write count・bytes）を safe-int 検証後に厳密な 0 として検証した。全ゼロ成功と各フィールドの 1、bool、float、負数、SAFE_INT 超過を再計算済み binding で追加検証した。
- 検証: compileall、関連 5 suites 229 passed、diff check。GPU、service、authorization、actual execution は未実施。
- 次の行動: 親エージェントが commit を統合し、再度独立監査を確認する。
