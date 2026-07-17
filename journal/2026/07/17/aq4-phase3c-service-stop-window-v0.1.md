# AQ4 Phase 3c: service stop window v0.1

## 前回の要点

- 1回目のPhase 3c試行（commit `53e12823`、journal `aq4-phase3c-r9700-window-execution-v0.1.md`）では、R9700 architecture guard（`gfx1201`、BDF `0000:47:00.0`）とR9700-only health telemetryは成功したが、AQ4本番gateway MainPID `1218698` が `/run/ullm/r9700.lock` を保持していた。`flock -n` は失敗し、service操作・GPU kernel起動・retryをせず安全停止した。
- 2026-07-17にユーザーから、`ullm-openai.service` の一時停止を含む今回だけの明示承認を得た。ただし既存runbookのR9700-only、single-execution、no-retry、evidence保存、P3 harness隔離、既存regular fileだけをlockとして使う契約は維持する。
- live baselineは、service `active/running`、MainPID `1218698`、worker `1218815`、`NRestarts=0`、`ExecMainStartTimestamp=Thu 2026-07-16 18:14:24 JST`、active manifest SHA-256 `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`だった。lock inode `848086` のFLOCK WRITE holderはgateway MainPIDだけだった。

## 今回の変更点

### Runbookと準備

- commit `a03fa06f`で、明示承認時だけ適用するservice一時停止・復旧手順をrunbookへ追加した。停止前snapshot、non-creating nonblocking lock probe、guard/telemetry/traceの順序、single `systemctl start`、Docker bridge内の`/healthz`/`/readyz`、worker/GPU/KFD/manifest確認、復旧失敗時の追加操作禁止を明記した。
- 実機unitは`RuntimeDirectory=ullm`かつ`RuntimeDirectoryPreserve=no`だった。07/15 journalの既知事実どおり、stop後にsystemdが`/run/ullm`を削除するため、lock leafがmissingならfree lockではなく取得失敗として扱うことをrunbookに明記した。`mkdir`、`touch`、`install`、lock作成・修復、P3 harnessの`prepare_lock_substrate`は使っていない。
- 最初のservice driver wrapperは引数処理不備で`systemctl stop`より前に終了した。直後のread-only再確認でMainPID/active stateが不変だったため、service/GPU windowは消費していない。この準備証跡を`service-stop-window-v0.1/`に保持し、上書きせずcommit `e9d3b56a`で実service windowのleafを`service-stop-window-v0.2/`へ分離した。
- v0.2の非GPU準備は成功した。frozen tooling diff、host-only HIP guard build、`cargo build --release -p ullm-engine --bin ullm-aq4-differential-trace --bin ullm-aq4-layer0-family-isolation`、CPU input identity、CPU referenceをservice停止前に完了した。CPU input identity/reportはともに`valid`、CPU stage streamは`24,692,172` bytesだった。

### 単回 service window

| step | UTC (JST) | 結果 |
| --- | --- | --- |
| stop開始・完了 | `2026-07-17T02:58:09+00:00` (`11:58:09 JST`) | `systemctl stop ullm-openai.service` は一回で成功（exit `0`） |
| post-stop lock probe | `2026-07-17T02:58:09.384318+00:00` | `/run/ullm/r9700.lock` は `ENOENT`。`O_CREAT`なしの`O_RDWR|O_NOFOLLOW`/`LOCK_EX|LOCK_NB` probeは取得不可 |
| trace | — | **未起動**。lock path absentをfreeと扱わず、待機・再probe・lock作成・GPU traceを行わなかった |
| start開始・完了 | `2026-07-17T02:58:09+00:00` (`11:58:09 JST`) | `systemctl start ullm-openai.service` は一回で成功（exit `0`） |
| health/readyとowner復旧確認 | `2026-07-17T02:58:18.588239+00:00` (`11:58:18 JST`) | 8回目のread-only pollで`/healthz`=`{"status":"ok"}`、`/readyz`=`{"status":"ready"}`。post-restore schemaは`valid` |

サービス停止からhealth/ready復旧までの所要時間は約9秒だった。`systemctl restart`、追加のstop/start、kill、rm、lock強制解放、manifest変更、V620 query、P3 harness操作は行っていない。

### 復旧後の状態

- serviceは`active/running`、MainPID `889726`、worker PID `890607`、`NRestarts=0`。明示的stop/startなのでPIDの更新は想定内であり、restart counterは停止前値から増えていない。
- active manifest SHA-256は停止前後とも`feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`で完全一致した。
- `/run/ullm/r9700.lock`は復旧後にregular file、mode `0600`、inode `889811`として再作成され、FLOCK WRITE holderは新gateway MainPID `889726`だけだった。
- workerは`/dev/kfd` FDを持ち、R9700 BDF `0000:47:00.0`だけへの`amd-smi process --gpu ... --general --json`はowner `[890607]`を返した。workerだけのKFD path `/sys/class/kfd/kfd/proc/890607/vram_51545` は`7,351,832,576` bytesだった。
- healthはhost直結ではなく、既存`open-webui` container network namespaceから`http://172.20.0.1:8000/healthz`と`/readyz`を確認した。これはdeploy/配下と過去journalで確立済みの実port/pathである。

### R9700/H9、段階比較、仮説

- 今回のservice windowではlock probe失敗後にrunbook規約どおりR9700 architecture guard、GPU health telemetry、traceを起動していない。したがって今回のguard/telemetryは**未実施**であり、V620を含む対象外GPUには触れていない。
- 前回試行のR9700 guardは`valid`（filtered HIP ordinal 0=`gfx1201`、BDF `0000:47:00.0`、AMD-SMI device ID `0x7551`）で、H9 telemetryはECC/UMC ECC/bad page=0、温度36--37 C、unthrottled、firmware/driver不変だった。今回のlock failureはGPU負荷前に起きたため、これを新しいH9観測としては使わない。

| stage | relative L2 | cosine | max abs | 判定 |
| --- | ---: | ---: | ---: | --- |
| `qkv_dequant_row_scale` | 未測定 | 未測定 | 未測定 | lock ENOENTにより測定無効 |
| `z_dequant_row_scale` | 未測定 | 未測定 | 未測定 | 同上 |
| `recurrent_gate` | 未測定 | 未測定 | 未測定 | 同上 |
| `recurrent_beta` | 未測定 | 未測定 | 未測定 | 同上 |
| `recurrent_state_after` | 未測定 | 未測定 | 未測定 | 同上 |
| `recurrent_output` | 未測定 | 未測定 | 未測定 | 同上 |
| `attention_residual` | 未測定 | 未測定 | 未測定 | 同上 |
| `post_norm` | 未測定 | 未測定 | 未測定 | 同上 |
| `mlp_activation` | 未測定 | 未測定 | 未測定 | 同上 |
| `layer_output` | 未測定 | 未測定 | 未測定 | 同上 |

- 最初の有意乖離stageは特定不能である。`<=1e-5`、`1e-5〜1e-3`、`1e-3〜1e-2`、`>1e-2`のどの数値分類にも入れない。
- H5（GPU kernel固有バグ）は**判定不能**。GPU kernelを一度も起動していないため、支持・否定のいずれにも使えない。H9についても今回の実負荷結果は判定不能である。

### 実行した検証とevidence

- `git diff --quiet <frozen trace tooling commit> -- ...` と index diff check: 成功。
- host-only `g++` build: 成功。guard binaryはbuildのみで、lock failure branchでは実行していない。
- CPU fixture verifier: `{"cases": 3, "status": "valid"}`。CPU input identity/report=`valid`。
- non-creating lock probe: `ENOENT`を`service-window-lock-after-stop.json`へ記録。`gpu-trace/`、`kernel-stages.f32le`、`cpu-gpu-stage-compare/`が存在しないことを確認した。
- post-restore `systemctl show`、Docker namespace health/ready poll、targeted R9700 AMD-SMI process、target worker KFD vram、manifest SHA、lock holder: すべて成功し`service-window-post-restore.json`=`valid`。
- 主なevidenceは `benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.2/` に保存した。これはlarge binaryを含む実行evidenceのため、既存方針どおりworktreeに保持し、今回のdocumentation commitにはstageしない。

## 次の行動

- この承認済みservice windowはlock ENOENTで消費済みであり、同一windowでのtrace retryは行わない。
- 現行契約のままserviceをstopすると`RuntimeDirectory` cleanupがlock pathを削除する。traceを可能にするには、lockをservice lifecycle外へ移すか、stop後の新規lock substrateを明示的に設計・承認する必要がある。これは既存regular-file-only契約の変更であり、今回のPhase 3cやPhase 4 fixには進めない。
- serviceは正常に復旧しているため、復旧対応は不要である。追加windowまたはlock lifecycle設計に関する次の判断を待つ。
