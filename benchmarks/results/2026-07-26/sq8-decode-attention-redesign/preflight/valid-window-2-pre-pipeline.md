# Valid window 2 — pipeline preflight

- Timestamp: `2026-07-26T22:22:33+09:00`.
- Required process check: no matching actual worker, measurement, or llama
  process. A long-lived prompt-writing shell contains the literal request text
  and is a `pgrep -af` false positive; it was excluded only after confirming
  the exact process names below and the GPU process table.
- `systemctl is-active ullm-openai.service`: `failed` (therefore no serving
  worker was running).
- `amd-smi process -G -g 2 --json`: `No running processes detected`.
- Exact process-name check for `llama-bench`, `llama-server`,
  `ullm-sq8-r9700`, and `ullm-aq4-worker`: empty.
- R9700 telemetry: edge `40 C`, hotspot `41 C`, memory `40 C`, socket power
  `14 W`, throttle `UNTHROTTLED`; reported idle clocks were gfx `1193 MHz`
  and memory `96 MHz`.

This is the start condition for only the remaining GQA-grouped, pipelined,
tile-20 full-model run.  It matches the 40/41/40 C starts used by the prior
valid variants.
