# AQ4 Phase 3c: amd-smi 絶対path化 v0.1

## 前回の要点

- `RuntimeDirectoryPreserve=yes` により、`ullm-openai.service` の stop/start をまたいでも `/run/ullm/r9700.lock` が同一 inode の regular file として存続することは実証済みである。
- 直近の service-stop window では、R9700 の HIP guard は `gfx1201` / `0000:47:00.0` を確認したが、続く ASIC cross-check が `runuser` 配下で bare `amd-smi` を解決できず未完了となった。trace と telemetry は開始せず、service は正常復旧済みである。
- 07/16 に停止した P3 harness の lock、root、artifact、環境変数には触れない。

## 今回の変更点

- `runuser -u homelab1 -- /usr/bin/env` の実測 PATH は `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin` であり、`/opt/rocm/bin` を含まないことを確認した。一方、AMD-SMI の実体は実行可能な `/opt/rocm/bin/amd-smi` である。
- 新規 `tools/run-aq4-phase3c-r9700-guard.py` を追加した。root から `runuser -u homelab1 -- /usr/bin/env HOME=/home/homelab1 HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 ...` を固定して、HIP identity、同じ BDF だけへの ASIC cross-check、ECC/clock/power/temperature、bad page、driver/IFWI、firmware を read-only で収集する。AMD-SMI は全て `/opt/rocm/bin/amd-smi` の絶対pathを使う。
- HIP で返った BDF が `0000:47:00.0`、`gfx1201`、可視台数 1、filtered ordinal 0 であることを検証してからだけ、AMD-SMI をその BDF に実行する。ASIC 側では `gfx1201`、PCI device ID `0x7551`、non-empty name、BDF 一致を検証する。V620 を列挙・照会する command は含めない。
- `docs/plans/aq4-phase3c-gpu-window-runbook-v0.1.md` の ASIC cross-check、H9 telemetry、復旧時 process 確認、preflight を絶対pathに更新した。PATH を拡張して解決する手順にはしていない。
- `pytest -q tests/test_aq4_phase3c_r9700_guard.py tests/test_aq4_phase3c_stage_tooling.py` は 5 passed、`python3 -m py_compile tools/run-aq4-phase3c-r9700-guard.py` と `git diff --check` は成功した。

## 次の行動

- service、systemd、active manifest、lock を変更せず、host-only で HIP guard binary を build してから、この新しい guard script を同一 `runuser` 経路で複数回リハーサルする。
- HIP/ASIC/health telemetry が安定して成功した場合だけ、既存 evidence を上書きしない新しい output leaf と一回だけの stop/start を使う Phase 3c window driver を準備する。
