# AQ4 Phase 3c Register-BM8 load admission / v0.7 preparation v0.1

## 前回の要点

- Phase 3c v0.6 は stop → lock → R9700 guard → trace → post guard → service 復旧を完走し、service 停止は約4秒だった。`service-stop-window-v0.6-complete-guard-set/gpu-trace.stderr` は layer 0 load 中の `unsupported backend operation Aq4MatvecBatch for phase ColdPrefill` を記録した。
- Phase 3b の audit どおり、実production M=1 path は `Qwen35Aq4Session::dispatch_prefill_chunk` の長さ1分岐から `dispatch_token_for_phase` を使う（`crates/ullm-engine/src/qwen35_aq4_session.rs:575-608`）。trace も `with_prefill_chunk_tokens(1)` を固定している（`crates/ullm-engine/src/bin/ullm-aq4-differential-trace.rs:1246-1250`）。
- したがって今回の修正範囲は trace tool、window driver、guard contract、runbook に限定し、production runtime/backend registry の実装は変更しない。

## 今回の変更点

### 根本原因の分類

- a は**実行 dispatch に限って**該当する。M=1 token は batch execution を使わず、`Aq4MatvecBatch` は実行時には不要である。
- b は不該当である。trace binary 自体が batch prefill に逸脱していたわけではない。
- c の「backend registry に未登録」は不該当である。通常の full-model loader が `resolve_aq4_batch_plans` で M=2..=128 の plan を eager resolve する（`crates/ullm-engine/src/aq4_package_runtime.rs:272-306`）。gfx1201/group16/適合行列には registry が M=2..7 の `HipAq4MatvecBatch` と M=8..128 の `HipAq4RegisterBm8` descriptor を登録する（`crates/ullm-engine/src/backend_operation_registry.rs:4454-4541`）。
- v0.6 では Batch guard は `=1` だった一方、Register-BM8 guard を unset していた。そのため normal loader の ColdPrefill plan admission が失敗し、generic operation kind として `Aq4MatvecBatch` が報告された。正しい修正は Batch guard を外すことではなく、`ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL=1` を17番目の load-admission guard として加えることだった。

### trace tool / driver / runbook

- trace tool commit `a7cb46e252b2f1a3e045278fe112bce21af05d32` は required guard を16→17にし、BM8 を disallowed set から除外した。M=1 dispatch と eager loader admission の違いをコメントと unit test で固定した。
- `tools/run-aq4-phase3c-service-window.sh` は同 commit を固定し、BM8 を trace child の `TRACE_ENV` と pre-stop JSON expected set に加えた。`linear_stage_guard.required_environment` について、件数だけでなく全17 guard の全値が `"1"` であることも assert する。script は executable（mode `0755` in git）である。
- `docs/plans/aq4-phase3c-gpu-window-runbook-v0.2.md` を追加し、v0.6 を上書きせず v0.7 の CPU-only 準備、17 guards、root-only rehearsal、最終 single window command を固定した。
- runtime/backend registry files は差分なしであることを `git diff a7cb46e... -- crates/ullm-engine/src/{qwen35_aq4_layer_runtime.rs,qwen35_aq4_model_runtime.rs,aq4_package_runtime.rs,backend_operation_registry.rs}` で確認した。

### v0.7 CPU-only preparation evidence

新規 final root は次であり、既存 evidence は変更していない。

```text
/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.7-register-bm8-load-admission
```

- host-only `query-hip-device-identity` を buildしただけで、実行していない。SHA-256 は `e85043b1bc1812a1b0ebcba31fcfa0bff5402be348d713a37f44643d9885175d`。
- `trace-binary-staging-verify-pre-stop.json` は `status=valid`、trace tooling commit一致、staged trace mode `0555`、`nlink=1`、SHA-256 `9660e49ef9b9beaab80ff218ed893426cc81d7c8aa3d255a7239e7e16a959cce` を確認した。
- `trace-guard-diagnostic-preflight.json` は `status=valid`。top-level required list と linear-stage guard map の双方で、17 guards が期待どおり全て `=1` であることを確認した。この `--print-phase3c-trace-guard-requirements` は HIP runtime/context/stream/kernel を作らない。
- `cpu-input-identity.json` は `status=valid`（3 cases）。`cpu-reference/aq4-report.json` は `status=valid`、`device=cpu:0`、`cpu-stages.f32le` は 24,692,172 bytes、stage summaries は28件である。
- CPU-only test は `cargo test -p ullm-engine --bin ullm-aq4-differential-trace -- --test-threads=1` の14 passed、`bash -n tools/run-aq4-phase3c-service-window.sh`、`pytest -q tests/test_aq4_phase3c_service_window_driver.py tests/test_aq4_phase3c_stage_tooling.py` の3 passed である。既存の runtime C++ subobject-linkage warning 以外の failure はない。
- `sudo`、service/lock 操作、R9700 HIP guard の実行、`amd-smi`、GPU trace は実行していない。V620、07/16停止中 P3 harness の lock/root/artifact/environment にも触れていない。

## 次の行動

親エージェントは、CPU-only evidence を確認後に service 稼働中で以下の create-new rehearsal を一度行う。これは今回実行していない。

```bash
sudo /usr/bin/python3 /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-r9700-guard.py \
  --output /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/guard-chain-rehearsal-v0.7-register-bm8-load-admission/attempt-1 \
  --guard-bin /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.7-register-bm8-load-admission/query-hip-device-identity \
  --health-phase pre-window
```

rehearsal が valid であることを確認してからだけ、次を親エージェントが直接実行する。`OUT_DIRECTORY` と `HIP_GUARD_BINARY` はいずれも絶対pathであり、driver の実行 bit は確認済みである。これは v0.7 の**唯一の** service-stop window である。

```bash
sudo /home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-phase3c-service-window.sh \
  /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.7-register-bm8-load-admission \
  /home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.7-register-bm8-load-admission/query-hip-device-identity \
  --confirm-single-window
```

driver の pre-stop staging verify または17 guard JSON assertion が失敗した場合は service を停止せず終了する。成功後の trace 結果は v0.6 evidence と混ぜず、この v0.7 root だけで評価する。
