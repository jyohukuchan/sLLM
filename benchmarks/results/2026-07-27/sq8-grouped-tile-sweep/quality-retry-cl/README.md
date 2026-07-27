# CL single-window retry — incomplete

This directory records the one service window consumed by CL at 11:15 JST.
It is not a tile-128 quality result.

The runner stopped `ullm-openai.service`, acquired `/run/ullm/r9700.lock`,
and skipped the already-recorded speed benchmarks.  Its first direct numeric
capture failed before GPU runtime initialisation:

```
failed to create new decode oracle capture directory .../numeric/direct/oracle:
No such file or directory (os error 2)
```

The initial fix had removed both the pre-existing capture target and its
needed parent.  Inspection of `sq8_ck_serving` established that the argument
itself is the new directory: the correct target is `numeric/<route>`, whose
parent `numeric/` already exists.  The harness is now corrected accordingly;
it neither precreates `numeric/<route>` nor appends `/oracle`.

Because the one window had already been consumed, no second lock/service
window was taken.  Thus direct/tile-20/tile-128 numeric comparisons, all eight
candidate generations (including `javascript_debug_extended`), and the MoE
physical runner are **not run** in this retry.  No quality conclusion or
`split_vs_direct` value is inferred from this failed setup attempt.

The active manifest remained
`a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd`.
After lock release, the production service was started once and is
`ActiveState=active`, `NRestarts=0`.  The initial restore probe raced gateway
readiness (`container_transport`); the bounded retry in
`service/restore-response-retry.json` received HTTP 200 with `restored`.
