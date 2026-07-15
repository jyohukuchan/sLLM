# P3 profile quiet window v9

## 前回の要点

- Resident profile の binding-v5、execute binding-v5、ready-v5、profile-ready-v5、および各 dry-run-v5 を sealed root として downstream cascade した。
- Actual 前の quiet-window は read-only とし、actual 実行、GPU workload command、service 操作を禁止した。

## 今回の変更点

- v5 境界に対して、最大 900 秒、nominal interval 5 秒、27 連続 clean かつ 130 秒以上の rolling observation を実施した。
- Rolling 自体は 27 連続 clean、`172.370043782` 秒、reset 0、monitor elapsed `186.039144114` 秒で完了した。PTS diagnostic identity の change count は 0 だった。
- Final HEAD は `35b5de98cf88eb7e06479e0d021dc99505208525`、index tree は `e10cb8fe10026646d636c687b257e0eab77de77c` だった。
- Relevant set は sealed root 6 個、regular file 43 個だった。byte aggregate は `9160cbb9003cea1bb8589cb286ea91a0e3244078ddfdbd49b2339e6a826a8ceb`、no-follow identity aggregate は `6f91f077e08aed8b23abaed6ea043eb9d35c13f42c8fd96f69fdd63c08880294` だった。
- Service identity、worker PID/hash、lock inode/holder、formal container-namespace health、exclusive AMD/KFD ownership は固定された。Blocking identity は baseline、final sample、confirmation で `8b80c5226581ce76205907e79fe35093796668f62622e8453c51f168101ed7e9` に一致した。
- Fresh profile-v5 の execute output、execute evidence、maintenance evidence、diagnostic capture root とその 3 leaves は confirmation まで absent だった。
- Strict QA provenance は 12/12、sealed root 6 個の `SHA256SUMS` はすべて通過し、confirmation sample と start/end formal health identity も通過した。
- 最終判定は fail-closed の `NO-GO` だった。原因は collector が canonical test node ID を 3 件誤指定し、pytest が `not found`、return code 4、`no tests ran` になったことだった。Actual lane の状態変化や blocking reset は原因ではない。
- Evidence SHA-256 は `446f30ec0683915e8dd23b3e9768d71ebe198024ac6401d0c7b82e1d796f82fc`、evidence `SHA256SUMS` SHA-256 は `a8f195e67c4233b476245db3d372cbb9a05a6e443303577c6d203b39751dad6e`、collector source SHA-256 は `2f4613668088ecb1eb0aaae2457052a9e536195f1284761bf70ebb111f991a82` だった。
- Safety fields は `read_only=true`、`actual_executed=false`、`gpu_command_executed=false`、`service_touched=false`、`secret_material_recorded=false` だった。
- Evidence root は mode `0555`、`quiet-window.json` と `SHA256SUMS` は mode `0444`、nlink 1 で固定した。

## 次の行動

- v9 は operator authorization に使用せず、canonical selector を `pytest --collect-only` で事前検証して exact node IDs と selected count を固定した v10 を別 lane で再走する。
- v10 が GO になっても operator authorization と actual execution は別の明示的な段階として扱う。
