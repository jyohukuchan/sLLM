# Owned BQ service window

`events.tsv` records one bounded lifecycle from 2026-07-27 01:06:04 through
01:09:53 JST: one `ullm-openai.service` stop, current AQ4 trace, three
isolated loopback gateways, one SQ8 comparison, and one successful restore.
`post-window-service.txt` records `ActiveState=active`, `SubState=running`,
and `NRestarts=0` after restoration.

The window never wrote `/etc/ullm/served-models/active.json`, never invoked a
promotion or rollback tool, and records the same active manifest SHA-256 at
pre-stop, pre-restore, and post-window.  The raw output directories named in
`events.tsv` contain the actual current AQ4 trace and generation captures.
