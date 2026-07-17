# AQ4 production prefill/decode optimization P0/P1

## 前回の要点

- AQ4 production prefill/decode optimization plan v0.1は、active productを変更せずにP0でidentity/rollbackを固定し、P1でtrace、独立validator、benchmark/evidence、read-only bottleneck auditを作る順序を定めていた。
- 既存release bundleのrollback environment hashはlegacy `/etc/ullm/openai-gateway.env` を指し、現在のsystemd drop-inはmanifest-mode `/etc/ullm/openai-gateway-manifest.env` を使っている。

## 今回の変更点

- `capture-aq4-production-p0.py`でactive manifest、worker、package、tokenizer、Git、systemd、Gateway、worker、OpenWebUI、GPU/driver/power conditionを非秘密artifactへcaptureした。
- current rollback bindingをmanifest-mode environmentへ再bindingした。source bundleとの差分は、served-model drop-inによるlegacy environmentからmanifest-mode environmentへの切替として記録した。active manifest/serviceは変更していない。
- `execution_trace.rs`とproduction-executor-record仕様を追加し、4 MiB、privacy、canonical digest、atomic publishの境界をengine側へ追加した。
- `produce-production-execution-trace.py`、`validate-production-execution-trace.py`、`run-aq4-production-performance-matrix.py`、`validate-aq4-production-optimization.py`、`audit-aq4-production-bottlenecks.py`を追加した。
- P1 production smokeでは、active manifestと同じresident worker binary/packageを別プロセスで一件実行した。128-token cold prefillと1-token decodeの実phase幅、192件のload-time operator resolution、request-terminal audit、R9700 VRAM observer、reset完了を含むbounded facts sidecarを生成した。
- `production-trace-live-v4-verified.json`と`production-executor-record-live-v4.json`をSHA bindingし、独立validatorで`status=valid`、`independent_validation=valid`、`promotion_eligible=true`を確認した。サービスの再起動・設定変更・active manifestの変更は行っていない。

## 次の行動

- P2で現active identityのbaselineを同じrun rootとidentity bindingで凍結する。R9700実測はGPU queueを直列化し、P3以降のcandidate比較へ進める。
