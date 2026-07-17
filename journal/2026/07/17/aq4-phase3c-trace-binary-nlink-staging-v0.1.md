# AQ4 Phase 3c trace binary nlink=1 staging v0.1

## 前回の要点

- `service-stop-window-v0.4-absolute-amd-smi-rehearsed` では、R9700 guard chainとH9 telemetryは通過したが、Cargo release outputの `target/release/ullm-aq4-differential-trace` が `nlink=2` だったため、trace binary自身のidentity contractがGPU kernel起動前にfail-closedした。
- この `nlink=2` はCargoが`deps/`側とのhard linkを作る正常な挙動であり、buildやlock、R9700 guardの異常ではない。serviceはwindow後に正常復旧済みである。
- 07/16のSQ8前例は、Cargo outputを直接使わず、SHA-256を維持したcreate-newの`0555`/`nlink=1` copyを実行artifactにする方式である。

## 今回の変更点

- `tools/stage-aq4-phase3c-trace-binary.py`を追加した。Cargo trace binaryをcontent copyで新規staging directoryへ作成し、source/staged SHA-256、mode、nlink、device/inodeを`staging-receipt.json`へ記録する。`SHA256SUMS`を作成後、staged binaryをregular file、mode `0555`、`nlink=1`としてread-only verifyする。既存stageの上書き、symlink、hardlink、`mv`を使う代替はfail-closedで拒否する。
- `tools/run-aq4-phase3c-service-window.sh`は、`OUT/trace-binary-staging/ullm-aq4-differential-trace`だけをtraceとして実行し、service停止前にstaging receipt/SHA/mode/nlinkを再検証するよう変更した。staging contractが失敗した場合はserviceを停止しない。
- runbookを新しい`service-stop-window-v0.5-nlink-staged` leaf、CPU-only staging preflight、trace binaryの固定SHA/nlink検証へ更新した。`ullm-aq4-layer0-family-isolation`には`current_exe()`/nlink identity guardがないことを確認し、CPU reference binaryはstaging対象から除外した。
- 新規stagerのhardlink切断、SHA保持、create-new拒否、検証失敗をCPU-only testで確認し、driver source testもstaging contractを確認するよう拡張した。

## 次の行動

- serviceを停止せず、固定trace tooling commitでrelease buildを行い、新しい`service-stop-window-v0.5-nlink-staged` evidence rootにstaging copyを生成する。sourceとstaged copyのSHA-256、mode、nlinkを記録・検証する。
- service稼働中にR9700 guard chainとstaging verifyを複数回リハーサルする。R9700だけを対象にし、V620、P3 harness、service/systemd/manifestには触れない。
- すべてが安定して成功した場合だけ、更新済みdriverによるservice-stop windowを一回実行する。trace内で失敗した場合は再試行せず、直ちにservice復旧結果を優先して記録する。
