# R9700 contention record before WMMA retry

At `2026-07-26T21:47:14+09:00`, a gateway restart failed to acquire
`/run/ullm/r9700.lock` (`BlockingIOError: [Errno 11] Resource temporarily
unavailable`).  At `21:47:25+09:00` the unit was therefore `failed`,
`MainPID=0`, `Result=exit-code`.

Read-only inspection at `21:47:xx`--`21:53:48+09:00` showed that PID `3040086`
(the concurrent BH measurement owner) held fd 9 on that lock after deliberately
stopping the service.  The R9700 had no candidate process from this task.

No lock was released, no process was signalled, no service start/stop was issued,
and no GPU workload was launched by this task while the foreign exclusive lock
was held.  The WMMA retry must start only after that owner releases the lock and
the service has returned to a defined `active` state; this observation is not a
WMMA measurement window.

At `2026-07-26T21:58:51+09:00`, the service's automatic retries had reached
`NRestarts=3` and it was again `failed` while the same foreign lock holder
remained.  This task did not add a start attempt or reset the unit; doing so
would have competed with the active exclusive owner and risked the configured
start-limit budget.

At `2026-07-26T22:14:xx+09:00`, AMD SMI additionally reported a foreign
`ullm-gemma4-resident` process (PID `3171000`) using 12,250,320,896 bytes on
R9700.  This task again launched no GPU work.  A retry requires both that
process to disappear and the BH flock to be released.
