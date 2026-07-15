# P3 profile quiet window v11

## 前回の要点

- Quiet-window v10 は commit `7c77535e`、27 連続 clean、172.226505035 秒、reset 0 で GO になった。
- Operator manifest v5 は commit `7b2a6a29` で作成され、actual 前に absent でなければならない fresh outputs を 9 件に拡張した。

## 今回の変更点

- Current epoch の HEAD に対して、最大 900 秒、nominal interval 5 秒、27 連続 clean かつ 130 秒以上の rolling observation を実施した。
- 開始前に canonical selector を `pytest --collect-only` で検証し、exact node ID 4 件、selected count 4 を確認した。Collector self-test では fresh 9/9 absent、sealed roots 6 個、regular files 43 個を確認した。
- Final decision は `GO`、streak は 27 連続 clean、`172.270818469` 秒、reset 0、monitor elapsed `186.145818908` 秒だった。PTS diagnostic identity の change count は 0 だった。
- Final HEAD は `7b2a6a29046a4a4f0abdbeea73fdefbf8d74857c`、index tree は `b930015ba06d882cd4062e1a90e6e5f8e0a2661b` だった。
- Relevant set は sealed root 6 個、regular file 43 個だった。byte aggregate は `9160cbb9003cea1bb8589cb286ea91a0e3244078ddfdbd49b2339e6a826a8ceb`、no-follow identity aggregate は `6f91f077e08aed8b23abaed6ea043eb9d35c13f42c8fd96f69fdd63c08880294` だった。
- Service identity、worker PID/hash、lock inode/holder、formal container-namespace health、exclusive AMD/KFD ownership は固定された。Blocking identity は baseline、final sample、confirmation で `f3628d4e3949557fcd0d6173d0f6f7319eb583434af2deb259088280b18ecd0b` に一致した。
- Operator manifest v5 が固定する profile runner、launcher evidence、maintenance evidence、capture root、artifact/stdout/stderr、operator result、actual audit の fresh 9 は confirmation まですべて absent だった。
- Strict QA provenance は 12/12、sealed root 6 個の `SHA256SUMS` はすべて通過し、canonical exact 4 tests は `4 passed in 0.25s`、confirmation sample と start/end formal health identity も通過した。終了後の独立 readback でも六 SUMS、fresh 9、4 tests が通過した。
- Collector wrapper source SHA-256 は `f408ce46e3193517d331047bd84eae01f5152b0a40b853ba5549e74089847ece` で、入力 bytecode SHA-256 `8d761db5b2a990a19dedf74174c955b77973114bebe21d59556fa99d384251ab` を固定検証した。一時 wrapper は evidence 生成後に削除した。
- Evidence SHA-256 は `218358d35c24b4ba14778ebcbbbf7afd67623f0a7d8f97e0510450ca81f651a1`、evidence `SHA256SUMS` SHA-256 は `2c0aa6b97f741bfe8aa03769bea26008dfdde52035c55074e50d68d066bce486` だった。
- Safety fields は `read_only=true`、`actual_executed=false`、`gpu_command_executed=false`、`service_touched=false`、`secret_material_recorded=false` だった。
- Evidence root は mode `0555`、`quiet-window.json` と `SHA256SUMS` は mode `0444`、nlink 1 で固定した。

## 次の行動

- v11 GO evidence と operator manifest v5 を operator 側で独立に再検証し、actual execution は別の明示的な段階として扱う。
- この quiet-window lane では actual execution、GPU workload command、service 操作を行わない。
