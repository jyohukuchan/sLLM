# R9700 measurement service lifecycle

All GPU work in this evidence directory targeted AMD SMI GPU `2`, `gfx1201`, PCI `0000:47:00.0`. `llama-qwen35-udq4.service` was checked as `inactive` and `disabled` before each isolated measurement window and was never started.

| window | `ullm-openai.service` action/result | `llama-qwen35-udq4.service` record |
|---|---|---|
| 03:42:49–03:43:56 JST | stopped for timing/initial profiler attempt; restored `active`/`enabled`; first profiler attempt exited 134 before a valid selected trace | inactive / disabled before and after (`service-lifecycle-attempt-2.txt`) |
| 03:45:21–03:46:06 JST | stopped for the valid scoped decode trace; restored `active`/`enabled`; exit 0 | inactive / disabled before and after (`service-lifecycle-attempt-3-decode-profile.txt`) |
| 03:49:53 | short prefill preflight stop/restore; no model-profile result accepted from this window | checked inactive / disabled before the window |
| 03:50:23–03:52:15 JST | stopped for the valid scoped prefill trace. The automatic restart reached `start-limit-hit` at 03:52:03; after confirming no isolated profiler process remained, `systemctl reset-failed` followed by `systemctl start` restored the same service at 03:52:15 | checked inactive / disabled before the valid window; it remained so |

Final read-only verification at `2026-07-26T04:09:41+09:00`:

```text
ullm-openai.service:        active / running / enabled
llama-qwen35-udq4.service:  inactive / dead / disabled
```

The system journal records the stop/start timestamps and the start-limit recovery. No unit-file content was edited. `/etc/ullm/served-models/active.json` was not read or written as part of this work.
