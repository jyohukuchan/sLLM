# AQ4 Phase 3c complete guard rehearsal v0.1

## 前回の要点

- `811d4271a9ef92f3df4699f0ba8a1862525e2661` で固定full-model traceの16 required guardとCPU-only自己診断を追加し、`7ec382a9c9c2ddb23a5d9a5bd612bade623cfe2d` でservice-window driver/runbookを同じ集合へ更新した。
- 最終windowは新規 `service-stop-window-v0.6-complete-guard-set` rootだけを使い、同一window内のtrace再試行をしない。
- V620、07/16に停止したP3 harnessのlock/root/artifact/environmentにはアクセス・変更していない。

## 今回の変更点

- serviceを停止・再起動せず、release binaryを `ULLM_BUILD_GIT_COMMIT=811d4271…` でCPU-only buildした。staged copyは `trace-binary-staging-verify-pre-stop.json` で`status=valid`、trace tooling commit一致、mode `0555`、`nlink=1`を確認し、`SHA256SUMS`も成功した。
- staged binaryに、全16 required guardを`=1`、非対象34 `ULLM_REQUIRE_HIP_*`と9 branch selectorをunsetにした環境で `--print-phase3c-trace-guard-requirements` を実行した。`trace-guard-diagnostic-preflight.json` は`status=valid`、required 16件、linear-stage guard map 16件である。これはHIP runtime/context/stream/kernelを作らない。
- CPU-only package embedding identityは`status=valid`（3 cases）、CPU layer-0 stage streamは`aq4-report.json`の`status=valid`、`cpu-stages.f32le`は24,692,172 bytesで生成された。直後のローカル確認が存在しない`cpu-reference/manifest.json`を仮定して失敗したが、実際の既存contractは`cpu-reference/aq4-report.json`であり、raw CPU evidenceは変更せず正しいpathで再検証済みである。
- `sudo -n python3 tools/run-aq4-phase3c-r9700-guard.py ...` はこのsessionで `sudo: a password is required` となった。root-only wrapperは実行されず、service/systemd/manifestにも変更はない。
- 代替のread-only rehearsalとして、service稼働中にhost-only `query-hip-device-identity`を`HIP_VISIBLE_DEVICES=1` / `ULLM_HIP_VISIBLE_DEVICES=1`で直接実行し、返ったBDF `0000:47:00.0`だけへ `/opt/rocm/bin/amd-smi static --gpu ... --asic --bus --json` と4 telemetry commandを実行した。validated evidence `guard-rehearsal-direct-{2,3,4}/r9700-direct-guard-rehearsal-validated.json` は全3回 `status=valid`、`gfx1201`、PCI device ID `0x7551`、health `complete`（metrics/bad-pages/static/firmware全てexit 0・JSON parse可）である。V620を対象にするqueryはない。
- direct rehearsal 1はHIP JSON取得後のローカルBDF正規表現のescaping誤りで止まり、AMD-SMIを起動していない。existing raw HIP evidenceを残し、上書きせず新規2--4で再実施した。2--4の初期summaryには末尾literal `\\n` が残ったため、raw HIP/AMD-SMI/telemetryを変更せずcreate-newの`*-validated.json`で再検証結果を保存した。
- 最終read-only確認時、`ullm-openai.service`は`active/running`、MainPID `1228628`であった。

## 次の行動

1. service-stop windowはroot権限がなければ実行できない。passwordを共有せず、ユーザーが次の一回だけのroot commandを実行するか、このsessionに限定した適切なroot実行手段を提供するまで待つ。

   ```bash
   sudo /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-service-window.sh \
     /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.6-complete-guard-set \
     /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.6-complete-guard-set/query-hip-device-identity \
     --confirm-single-window
   ```

2. 実行後はtrace exit codeにかかわらずservice restore/healthを最優先で確認し、成功時だけchecksum・10 stage CPU/GPU比較を実行する。同一windowのtrace再試行はしない。
