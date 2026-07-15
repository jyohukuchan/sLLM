# P3 profile quiet window v10

## 前回の要点

- Quiet-window v9 の rolling state は 27 連続 clean、172.370043782 秒、reset 0 だったが、collector が存在しない canonical test node ID を 3 件指定したため fail-closed の `NO-GO` になった。
- v9 は commit `5518d913` で immutable evidence とし、operator authorization には使用しないことにした。

## 今回の変更点

- v10 開始前に canonical selector を `pytest --collect-only` で検証し、exact node ID 4 件、selected count 4 を確認した。同じ selector の preflight 実行も `4 passed in 0.25s` だった。
- v9 の compiled collector artifact を一時実行物として復元し、v10 output/schema と検証済み exact test tuple を runtime で固定した。collector artifact SHA-256 は `8d761db5b2a990a19dedf74174c955b77973114bebe21d59556fa99d384251ab` であり、一時実行物は evidence 生成後に削除した。
- 最大 900 秒、nominal interval 5 秒、27 連続 clean かつ 130 秒以上の rolling observation を実施し、`GO` になった。
- Final streak は 27 連続 clean、`172.226505035` 秒、reset 0、monitor elapsed `186.112446123` 秒だった。PTS diagnostic identity の change count は 0 だった。
- Final HEAD は `5518d9133000be3c08e32676eadf8df38c492c5e`、index tree は `a5cff162672876f41459a8ca5f9401442a275ab3` だった。
- Relevant set は sealed root 6 個、regular file 43 個だった。byte aggregate は `9160cbb9003cea1bb8589cb286ea91a0e3244078ddfdbd49b2339e6a826a8ceb`、no-follow identity aggregate は `6f91f077e08aed8b23abaed6ea043eb9d35c13f42c8fd96f69fdd63c08880294` だった。
- Service identity、worker PID/hash、lock inode/holder、formal container-namespace health、exclusive AMD/KFD ownership は固定された。Blocking identity は baseline、final sample、confirmation で `492e3d98190bad742eebf6a3de2f3e224981ef7f4987bc57daf9405b789c1a90` に一致した。
- Fresh profile-v5 の execute output、execute evidence、maintenance evidence、diagnostic capture root とその 3 leaves は confirmation まで absent だった。
- Strict QA provenance は 12/12、sealed root 6 個の `SHA256SUMS` はすべて通過し、canonical tests は `4 passed in 0.24s`、confirmation sample と start/end formal health identity も通過した。終了後の独立 readback でも六 SUMS と 4 tests（`4 passed in 0.25s`）が通過した。
- Evidence SHA-256 は `30e5cb3c1e443ce8f0eb49d76ce68d69a2519fba5c26b0381c37ffc07d57af84`、evidence `SHA256SUMS` SHA-256 は `028103e798e8ae47e33dcf9987cfe0d856939ddeb16360b508e9e221498e6ca6` だった。
- Safety fields は `read_only=true`、`actual_executed=false`、`gpu_command_executed=false`、`service_touched=false`、`secret_material_recorded=false` だった。
- Evidence root は mode `0555`、`quiet-window.json` と `SHA256SUMS` は mode `0444`、nlink 1 で固定した。

## 次の行動

- v10 GO evidence を operator 側で独立に確認し、operator authorization を別 commit、別の明示的な段階として扱う。
- この quiet-window lane では actual execution、GPU workload command、service 操作を引き続き行わない。
