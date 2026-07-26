# R9700 measurement-window and service record

## Windows

| window | status | service / GPU condition | disposition |
|---|---|---|---|
| 1 | invalid | `ullm-openai.service` resumed and an external measurement process overlapped the attempted timing | retained only in `../preflight/contaminated-window-1.md`; no result reported |
| 2 | valid | service stopped, R9700 process table empty before every valid run | all `valid-*` probe and full-model records are reportable |

The external `llama-bench` that appeared after the grouped measurement and
before the pipeline measurement was allowed to finish. The pipeline run did
not start until a later process check found the R9700 table empty. Its start
condition was 40/41/40 C and `UNTHROTTLED`; its post-run GPU table was also
empty.

## Restore

This task issued two service stops across the two attempted windows. Before the
single final restore, the verified service state was:

```text
Result=exit-code
NRestarts=3
ActiveState=failed
SubState=failed
Jul 26 21:58:28 ... Start request repeated too quickly.
```

Because the start-limit condition was explicit, the recovery followed the
lightweight-promotion policy's one-time exception:

```text
systemctl reset-failed ullm-openai.service
systemctl start ullm-openai.service
```

Each command was issued once. At `2026-07-26T22:24:45+09:00` the verification
reported:

```text
ActiveState=active
SubState=running
Result=success
NRestarts=0
```

`llama-qwen35-udq4.service` was not started; its final state remained
`inactive` and `disabled`.
