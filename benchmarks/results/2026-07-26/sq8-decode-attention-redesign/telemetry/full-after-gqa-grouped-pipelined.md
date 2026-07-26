# Post-run R9700 telemetry: grouped+pipelined tile 20

Immediately after the valid pipeline full-model runner completed:

```text
GPU process table: No running processes detected
temperature: edge 57 C, hotspot 58 C, memory 56 C
socket power: 16 W
throttle_status: UNTHROTTLED
reported clocks: gfx 1193 MHz, memory 96 MHz
```

The start condition is recorded in
`../preflight/valid-window-2-pre-pipeline.md`. This telemetry is operational
context only; it is not used to derive throughput or physical bandwidth.
