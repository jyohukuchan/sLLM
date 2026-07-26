# Attempt status

This directory is an **aborted pre-measurement service window**, not a timing
result.  It began at `2026-07-26T21:45:58+09:00` with the service active.  The
intentional stop left `ullm-openai.service` as `failed` / `MainPID=0` because
the gateway reports its worker shutdown as a nonzero exit.  The then-current
runner accepted only `inactive`, exited before R9700 isolation or any
candidate workload, and its EXIT trap restored the service to `active/running`
at `21:45:59+09:00`.

There is deliberately no `timing/` result.  The retry uses a distinct output
directory after updating the stop-state check to accept the documented
no-main-PID `failed` state.  This abort counts as one stop/start attempt for
service-window accounting, but as zero GPU timing windows and zero candidate
measurements.
