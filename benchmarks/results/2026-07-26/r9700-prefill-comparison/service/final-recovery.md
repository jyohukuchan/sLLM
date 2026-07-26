# Service recovery audit

This file distinguishes the one intentional benchmark isolation window from
the later service recovery needed to leave the machine restored.

| time (JST) | observed action/state | evidence |
| --- | --- | --- |
| 19:12:23 | Intentional stop of `ullm-openai.service`; it became inactive. | `stop.txt`, `post-stop-state.txt` |
| 19:12:23--19:59:57 | The only measurement window; all 15 conditions completed. | `../runner-complete.json` |
| 19:59:57 | Wrapper `sudo -n systemctl start` attempt returned 1 because the cached credential had expired. | `restore.txt` |
| 20:00:24 | systemd journal showed the service starting; initiator unconfirmed. | journal excerpt captured during final audit |
| 20:05:57 | Service stopped after gateway record `unexpected worker stdout EOF`. | journal excerpt captured during final audit |
| 20:06:13 | Approved explicit `systemctl start` issued to restore the inactive service. | final audit |
| 20:06:14 | systemd reported service started, MainPID 423448. | final audit |
| 20:08:11 | Gateway recorded another `unexpected worker stdout EOF`; systemd restarted it at 20:08:12. | `post-measurement-service-events.txt` |
| 20:09:06 | Another worker EOF; systemd then reported `start-limit-hit`. | `post-measurement-service-events.txt` |
| 20:09:50 | systemd showed a new start; initiator unconfirmed. | `post-measurement-service-events.txt` |
| 20:11:56 | Another worker EOF; systemd immediately restarted it. | `post-measurement-service-events.txt` |
| 20:12:42 | Another worker EOF; systemd restarted it at 20:12:43. | `post-measurement-service-events.txt` |
| 20:13:48 | Audit status: active/running/enabled, MainPID 954753, NRestarts=0. | final audit |
| 20:16:26 | Final pre-commit audit: still active/running/enabled, MainPID 954753, NRestarts=0; no new event since 20:12:43. | `final-status.txt` |
| 20:17:27 | Another worker EOF stopped the service. | `post-measurement-service-events.txt` |
| 20:19:18 | A normal start was rejected with `start-limit-hit`. | `post-measurement-service-events.txt` |
| 20:19:34--20:19:35 | Approved `reset-failed`, then one explicit start; systemd reported active. | `final-status-after-start-limit-recovery.txt` |
| 20:20:10 | Final audit: active/running/enabled, MainPID 1480646, NRestarts=0. | `final-status-after-start-limit-recovery.txt` |

No additional stop and no benchmark process occurred after 19:59:57.  Thus
there was one intentional isolation window, followed by a restoration audit.
The later worker-EOF/restart sequence occurred during post-measurement
service operation; its cause and the initiator of the 20:00:24 and 20:09:50
starts are unconfirmed.  After the 20:17:27 EOF, `StartLimitBurst=3` blocked
a normal start; the one approved `reset-failed` plus start restored service at
20:19:35.  `llama-qwen35-udq4.service` was again confirmed
inactive/dead/disabled with MainPID 0 after the final restoration and was
never started.

The complete filtered post-measurement event chronology is
[post-measurement-service-events.txt](post-measurement-service-events.txt).
Relevant excerpts are:

```text
2026-07-26T20:05:57+09:00 ... worker_fatal ... reason: unexpected worker stdout EOF
2026-07-26T20:05:57+09:00 ... ullm-openai.service: Deactivated successfully.
2026-07-26T20:06:13+09:00 ... Starting ullm-openai.service
2026-07-26T20:06:14+09:00 ... Started ullm-openai.service
2026-07-26T20:09:06+09:00 ... ullm-openai.service: Failed with result 'start-limit-hit'.
2026-07-26T20:12:43+09:00 ... Started ullm-openai.service
2026-07-26T20:19:35+09:00 ... Started ullm-openai.service
```
