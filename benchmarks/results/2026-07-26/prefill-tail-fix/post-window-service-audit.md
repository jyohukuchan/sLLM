# Post-window service audit

Audit time: 2026-07-26 22:00 JST.

This note is deliberately separate from `service-window.md`.  The latter
records the one prefill measurement window, which ended at 21:44:47 JST with
the task's single recovery `systemctl start` succeeding and the service
reported `active`.

After that window, the unrelated AQ4_0 gateway received requests, then its
worker reported stdout EOF.  Subsequent gateway starts at 21:47 and later
failed before worker launch with:

```text
WorkerBusy: another process owns the GPU singleton lock
```

At the first post-window audit, an unrelated
`sq8_0_paged_decode_steady_bench` process was observed using the R9700.  After
that process exited, AMD SMI reported no running R9700 process and
`/run/ullm/r9700.lock` was absent.  A single policy-shaped recovery was then
attempted: `systemctl reset-failed` followed by one `systemctl start`.  The
gateway still hit transient lock contention; systemd's automatic retries took
the counter to three and left the unit `failed`/start-limited.  No additional
manual retry was issued.

The exact holder of the short-lived `/run/ullm/r9700.lock` contention was not
captured, so its cause is **unconfirmed**.  This event is outside the isolated
prefill measurement interval and is not evidence about the SQ8_0 tail-fix
binary.  The active manifest was never changed and remains:

```text
c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4
```

At this audit, `llama-qwen35-udq4.service` remained `inactive` and `disabled`.
