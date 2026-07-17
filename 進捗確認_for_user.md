# Phase 3c GPU window 進捗

- RuntimeDirectoryPreserve=yes の専用 drop-in を追加し、daemon-reload のみで適用した。その後の唯一の service-stop window で、/run/ullm/r9700.lock は stop/start をまたいで同一 device/inode の regular file として存続し、no-create nonblocking flock も成功した。systemd 側の根本原因は修正・実証済み。
- R9700 HIP guard は gfx1201 / 0000:47:00.0 で成功したが、同一 BDF の ASIC cross-check は runuser の default PATH が amd-smi を見つけられず未完了になった。single-use 契約に従って trace・telemetry・比較は起動せず、再試行もしなかった。
- service は正常復旧済み（active/running、NRestarts=0、manifest SHA 一致、healthz/readyz 成功、worker/GPU/KFD/lock owner 再取得）。H5/H9 は未判定であり、追加 window には改めて明示承認が必要。
- 07/16 停止中 P3 harness と既存 evidence には触れていない。
