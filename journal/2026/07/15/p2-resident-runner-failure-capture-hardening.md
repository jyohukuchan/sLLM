# P2 resident runner failure capture hardening

## Outcome

- The runner now streams resident-driver stderr directly to a private `O_EXCL` file instead of an unread pipe.
- Driver stdout is binary, line-bounded to 1 MiB, and unexpected/non-JSON responses retain only byte counts, SHA-256, and a bounded secret-scanned tail.
- Immutable driver process evidence records spawn PID/PGID/time, protocol stage and counters, ready state, exit code or signal, stderr bytes/SHA-256, cleanup signals, descendant process-group state, and lock inode continuity.
- Stderr larger than 1 MiB retains only a 64 KiB tail. Secret markers suppress all stderr text retention.
- The process is started in a new session and cleanup escalates from protocol shutdown to process-group `SIGTERM` and `SIGKILL`.

## Offline verification

- `tests/test_run_aq4_p2_resident_batch.py`: 37 passed.
- Fault coverage includes early exit, 2 MiB stderr, signal termination, response timeout, invalid JSON, secret stderr, mid-run exit, protocol OOM, and a hanging descendant.
- Bundle validator: 36 passed.
- Launcher: 7 passed.
- Execute launcher: 58 passed.
- The historical actual-attempt-v5 output remains untouched. The next base one-case run ID, runner output, launcher evidence, and live-preflight binding were advanced together from v1 to unused v2 paths.
- Maintenance: all 76 passed, including the 7 fake execute-CLI cases, with no deselection.
- Combined runner, validator, launcher, execute-launcher, and maintenance regression: 214 passed.
- The v2 runner output and launcher evidence paths remained absent after all offline tests. The third fresh output is the next operator-selected maintenance evidence path and is checked by the operator manifest.
- All regenerated binding, execute-binding, base/profile ready, and base/profile dry-run `SHA256SUMS` verified.
- No actual execution, service operation, GPU access, HTTP request, or sudo command was performed.

## Trust-chain commits

- Runner: `4005c80e`
- Validator: `741616fd`
- Binding v4: `1d964602`
- Initial hardened launcher/execute/harness/ready chain: `79ce2aa1`, `4507b20b`, `d3d9eaee`, `a1547b6f`
- V2 output launcher: `4a7c0ed9`
- V2 execute binding: `e1fe6353`
- V2 maintenance harness: `7e597eb6`
- V2 ready/profile artifacts: `b5e85b60`
