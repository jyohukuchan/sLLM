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

### Step 1 実測（service停止なし、CPU-only）

- 固定trace tooling commit `5a0fb4c50476d5153ced22bd6847c2729bfdb975` に対する対象source/index diffがcleanであることを確認してから、`CARGO_BUILD_JOBS=1 ULLM_BUILD_GIT_COMMIT=5a0fb4c50476d5153ced22bd6847c2729bfdb975 cargo build --release -p ullm-engine --bin ullm-aq4-differential-trace --bin ullm-aq4-layer0-family-isolation`を成功させた。既知のC++ warning以外のbuild failureはなかった。
- Cargo source `target/release/ullm-aq4-differential-trace` はmode `0700`、`nlink=2`、size `3002128`、SHA-256 `835ca1cb15aba577ef72902af719451a91c55371cbff8c41444d8f343469f2a4`だった。これは期待どおりのhardlink状態である。
- create-new staging rootは `benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.5-nlink-staged/trace-binary-staging/` とした。staged `ullm-aq4-differential-trace` はmode `0555`、`nlink=1`、size `3002128`、device/inode `66306:10754708`、SHA-256はsourceと同じ `835ca1cb15aba577ef72902af719451a91c55371cbff8c41444d8f343469f2a4`である。
- `staging-receipt.json`と`SHA256SUMS`はいずれもmode `0444`/`nlink=1`であり、stagerのread-only verifyは`status=valid`、source/staged SHA一致、staged mode `0555`/`nlink=1`を返した。staging directory内で実行した`sha256sum -c SHA256SUMS`も2 entryとも`OK`だった。
- 最初にparent directoryから`sha256sum -c "$TRACE_STAGE_DIR/SHA256SUMS"`を呼んだため相対entryを解決できなかったが、artifactを書き換えず、staging directory内でのread-only検証を一回行って成功した。これはstaging copyやidentity contractの失敗ではない。
- GPU kernel、HIP guard、AMD-SMI、service/systemd/manifest、V620、P3 harnessにはこのStepで一切触れていない。

### Step 2 リハーサル結果（service稼働中）

- `sudo`のTTY別認証期限のため、`guard-chain-and-staging-rehearsal-v0.1`と`v0.2`はguard起動前に認証エラーで止まった。いずれもservice、GPU guard、AMD-SMIに未到達であり、既存partial leafは上書き・修復せず保持した。
- 新規leaf `guard-chain-and-staging-rehearsal-v0.3`で、rootの単一TTY境界から3回連続でguardを実行した。各attemptのstaging verifyは`status=valid`、staged traceはmode `0555`/`nlink=1`、guard summaryは`status=valid`、architecture guardは`gfx1201`、PCI BDF `0000:47:00.0`、PCI device ID `0x7551`、H9 telemetryは`complete`だった。
- guardの対象は各回とも `HIP_VISIBLE_DEVICES=1` → filtered ordinal 0 → `0000:47:00.0`だけである。V620の列挙・照会を行うcommandは使っていない。guardはidentity/AMD-SMI read-only queryだけで、HIP stream、device memory、kernel launchは行っていない。
- service read-only snapshotはリハーサル前後とも`active/running`、MainPID=`1128520`、`NRestarts=0`、`RuntimeDirectoryPreserve=yes`だった。service/systemd/manifestには変更を加えていない。

### Step 3 停止前CPU準備

- serviceを停止せず、staging rootをもう一度read-only verifyして`status=valid`、staged trace SHA-256 `835ca1cb15aba577ef72902af719451a91c55371cbff8c41444d8f343469f2a4`、mode `0555`、`nlink=1`を確認した。
- `tools/verify-aq4-layer0-package-embedding-fixture.py`は`{"cases": 3, "status": "valid"}`を返した。CPU-only `ullm-aq4-layer0-family-isolation --stage-stream-stdout`は成功し、`cpu-stages.f32le`を`24692172` bytesで生成した。
- この準備ではHIP visibilityと全HIP kernel required環境変数をunsetし、GPU kernel、AMD-SMI、service/systemd/manifest、V620、P3 harnessには触れていない。

## 次の行動

- 更新済みdriverによるservice-stop windowを一回だけ実行する。trace内で失敗した場合は再試行せず、直ちにservice復旧結果を優先して記録する。
