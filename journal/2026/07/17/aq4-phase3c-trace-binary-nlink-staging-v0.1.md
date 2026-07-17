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

### Step 3 単回service-stop window結果

#### 最優先: service復旧

- `ullm-openai.service`は`2026-07-17T04:30:34+00:00`に一回だけstopされ、`2026-07-17T04:30:36+00:00`に一回だけstartを呼び、`2026-07-17T04:30:37+00:00`に成功した。stop invocationからstart成功まで約3秒であり、restart、追加stop/start、trace retryは実行していない。
- restore後は`active/running`、MainPID=`1228628`、worker PID=`1228734`、`NRestarts=0`、`RuntimeDirectoryPreserve=yes`、manifest SHA-256=`feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`で停止前と一致した。workerの`/dev/kfd` FD、`/sys/class/kfd/kfd/proc/1228734/vram_51545`、R9700 BDF `0000:47:00.0`だけへのAMD-SMI process queryのowner PIDも確認した。Docker bridge namespaceのhealthz=`{"status":"ok"}`、readyz=`{"status":"ready"}`である。
- 最初のpost-restore collectorはPPID文字列の解析ミスで`service-window-post-restore.json`をinvalidとして残し、次のcollectorは有効状態を記録したが末尾literal `\\n`のためJSON parserに拒否された。いずれもservice操作やevidence上書きはしていない。`service-window-post-restore-v0.3-reconstructed.json`はv0.2の収集済みvalid dataだけからread-onlyで整形式再構成した最終receiptである。

#### lock、staging、R9700 guard

- stop後のno-create lock probeは`2026-07-17T04:30:34.383260+00:00`に既存regular file、mode `0600`、device/inode `26:889811`を確認し、`O_RDWR|O_NOFOLLOW|O_CLOEXEC`と`LOCK_EX|LOCK_NB`を取得・解放した。`create_flag_used=false`である。
- driverのstop前staging verifyを通過した後にstaged binaryが実行された。window後のread-only verifyでもSHA-256 `835ca1cb15aba577ef72902af719451a91c55371cbff8c41444d8f343469f2a4`、mode `0555`、`nlink=1`、inode `10754708`を維持している。したがって今回のfailureはCargo hardlink/trace binary identity contractのfailureではない。
- trace前後のR9700 guardはともにvalid、HIP filtered ordinal 0=`gfx1201`/`0000:47:00.0`、AMD-SMIは同一BDF/`gfx1201`/`0x7551`だった。V620を対象にするcommandは使っていない。

#### trace failureと比較不能の理由

- trace invocationは`2026-07-17T04:30:35+00:00`に一回だけ開始し、同秒にexit code `1`で終わった。stderrは`failed to load Qwen3.5 AQ4 linear layer 0: required backend operation runtime feature/guard is unavailable`である。`gpu-trace/` publish root、kernel stage stream、manifest、30 record比較は生成されていない。
- source inspectionではlinear layer loadが`HipLinearAttentionRecurrent`、`HipLinearAttentionQkvPrepare`、`HipAq4MatvecBatch`、`HipLinearAttentionQkvPrepareBatch`の4 featureを必須化する。一方、driverはrecurrentとQKV-prepare-batchをpolicyへ入れる環境変数は設定したが、`ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL=1`と`ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`を設定していない。不足した`HipLinearAttentionQkvPrepare`と`HipAq4MatvecBatch`がprovenにならず、このgeneric require_features failureになったと読むのが根拠のある結論である。
- capability probeはpolicyで有効なscratch HIP operationを実行してからfeatureをprovenにする実装なので、traceは完全な「GPU未接触」ではない。ただしproduction layer0 forwardと10 stage D2H traceには到達していない。single-use契約に従い、環境変数を追加して同一windowで再実行することはしていない。

#### H9 telemetry、仮説判定

- before: power `15 W`、throttle `UNTHROTTLED`、gfx/mem clock `905/96 MHz`、edge/hotspot/mem `38/39/36 °C`、ECC total/UMC correctable・uncorrectable・deferredはすべて`0`、bad pageはなし、perf level=`AUTO`。
- after: power `14 W`、throttle `UNTHROTTLED`、gfx/mem clock `49/96 MHz`、edge/hotspot/mem `38/39/36 °C`、ECC/UMC ECCはすべて`0`、bad pageはなし、perf level=`AUTO`。driver=`amdgpu 6.16.13`、IFWI=`SAPPHIRE RADEON AI 32GB` version `00158746`、firmware recordsもbefore/afterで同一だった。
- 10 stageすべて（QKV dequant、Z dequant、gate、beta、recurrent state/output、attention residual、post norm、MLP activation、layer output）は未測定であり、relative L2/cosine/max absのthreshold判定は適用不能である。最初の有意乖離stageは特定不能。
- H5（GPU kernel固有バグ）は判定不能である。H9（hardware要因）も、telemetryに異常はないが実production stage traceがないため判定不能である。H9を支持する直接evidenceは得られていない。

## 次の行動

- このwindowでtrace retry、feature guard修正、Phase 4以降のfix実装には進まない。次のGPU windowを検討するには、今回のruntime-feature guard不足を独立にreviewし、別途明示承認を得る必要がある。
