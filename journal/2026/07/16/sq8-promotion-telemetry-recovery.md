# SQ8 promotion telemetry recovery

## 前回の要点

- One-shot GPU promotion run failed closed because the request-scoped telemetry did not contain both a positive batch projection count and a positive pair projection count.
- Immutable failure evidence was retained under `/tmp/ullm-sq8-overlay-gpu-promotion-actual-ba3f02ee-362cfa9587b04197`.
- The capture error incorrectly reported `worker_not_started` even though the worker had completed and emitted a terminal `request_released` audit.

## 今回の変更点

- Root cause: the fixed request used 128 prompt tokens and `max_new_tokens=1`. The prefill executed SQ8 batch projections, but the first generated token was sampled from the prefill result, so no decode model step executed and the SQ8 pair count remained zero.
- The actual promotion request now requires exactly two generated tokens and disables EOS stopping for this evidence-only request. This guarantees one decode model step after the 128-token prefill without fabricating or inferring counters.
- The strict telemetry validator and all positive/zero thresholds remain unchanged.
- Output identity is now bound to exactly two tokens in both the maintenance runner and receipt writer.
- Capture error schema v3 preserves the raw, structurally valid pre-threshold telemetry object, its canonical-JSON SHA-256 binding to the expected promotion request ID, worker return code/signal/stderr, and a bounded `request_released` terminal summary.
- Terminal evidence is extracted after complete stderr collection even when the worker exits nonzero. Later audit, resource-observation, and package-validation failures retain the same telemetry and terminal evidence.
- The maintenance runner validates the terminal request ID's strict format and exact expected value. It independently recomputes the telemetry hash for both successful executor records and failed capture envelopes.
- The receipt writer independently recomputes the successful executor record's telemetry hash and validates the bound request ID before copying the binding into actual evidence.
- CPU/fake-worker coverage includes positive batch+pair execution evidence, batch-zero, pair-zero, telemetry hash and request-ID tampering, terminal request-ID mismatch, nonzero worker exit, and post-telemetry audit/resource/package failures.

## 検証

- `python3 -m py_compile` for the three modified tools: passed.
- Focused promotion/capture/receipt suite: 124 passed with one pytest process.
- Broader SQ8 and served-model suite: 226 passed, 5 subtests passed, 1 unrelated isolated-worktree path assertion failed. The failing profile contains the original checkout's absolute worker path while the test expects the isolated worktree path.
- `git diff --check`: passed.
- No GPU command, service mutation, or authorization action was performed.

## 次の行動

- Commit the telemetry-binding and failure-preservation hardening as a separate isolated change.
- Provide the new commit, tree, archive digest, changed-file inventory, and test results to the parent task.
