# Service and GPU window

## Window count and recovery

One isolated measurement window ran from 21:11:26 to 21:44:47 JST.
`ullm-openai.service` was already inactive before this task’s owned stop step,
so the wrapper explicitly adopted that inherited quiet state rather than
issuing a second stop.  It then performed the old/new numerical captures and
all five candidate timing conditions, and its EXIT handler issued one
`systemctl start` recovery.

| action | result |
| --- | --- |
| task-issued stop | none (`not-issued-inherited-inactive`) |
| isolated windows | 1 |
| restore `systemctl start` | one attempt, return 0 |
| post-restore state | `active` |
| `llama-qwen35-udq4.service` | remained `inactive` / `disabled` |

The event sequence is append-only in
[`service/window-events.tsv`](service/window-events.tsv).  At final audit,
`ullm-openai.service` was `active/running`, `NRestarts=0`, and the AQ4_0
worker was present.

## Active manifest and promotion decision

The active manifest SHA-256 before and after recovery is unchanged:

```text
c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4
```

It is the currently active AQ4_0 served-model manifest, not an SQ8_0 tail-fix
candidate manifest.  The task instruction requires reporting rather than
overwriting an unexpected active manifest, so no promotion tool was run and
`/etc/ullm/served-models/active.json` was never modified.

## GPU contention note

The required process preflight was captured before the window.  A separate
Gemma4 trace appeared during the 2048 **cooldown**, before the candidate
driver was launched.  The thermal gate held that condition until the foreign
trace had exited; manual process checks during the 2048 and 4095 driver runs
showed only the tail-fix driver.  The raw thermal streams remain the source of
record for the start gates and timed-process telemetry.
