# Thermal, clock, power, and throttle-status summary

The pre-process thermal gate was sampled immediately before every benchmark
process.  The timed-start column is the nearest one-second AMD SMI sample
after the driver's timed marker; it is not a hardware-synchronized
measurement.  Accordingly, it is evidence of the warm-up difference, not a
claim of identical timed-start temperature.  The raw telemetry is
`thermal-history.csv` and the per-process JSONL under `raw/`.

| prompt | condition | gate edge/hotspot/mem C | gate socket W | nearest timed marker edge/hotspot C | max edge/hotspot/mem C | max gfx MHz | max socket W | throttle strings |
| ---: | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| 128 | ullm-sq8_0-f32-kv-p128 | 38/38/36 | 15 | 44/61 | 44/61/44 | 3278 | 109 | THROTTLED, UNTHROTTLED |
| 128 | llama-cpp-q8_0-f32-kv-p128 | 40/41/40 | 13 | 44/61 | 44/61/44 | 3100 | 244 | THROTTLED, UNTHROTTLED |
| 128 | llama-cpp-q8_0-f16-kv-p128 | 40/41/40 | 7 | 44/62 | 44/62/46 | 3084 | 232 | THROTTLED, UNTHROTTLED |
| 512 | ullm-sq8_0-f32-kv-p512 | 40/41/40 | 16 | 46/63 | 51/69/52 | 3368 | 404 | THROTTLED, UNTHROTTLED |
| 512 | llama-cpp-q8_0-f32-kv-p512 | 40/41/40 | 16 | 46/64 | 46/64/48 | 3125 | 277 | THROTTLED, UNTHROTTLED |
| 512 | llama-cpp-q8_0-f16-kv-p512 | 40/41/40 | 14 | 46/63 | 47/63/48 | 3143 | 287 | THROTTLED, UNTHROTTLED |
| 1024 | ullm-sq8_0-f32-kv-p1024 | 40/42/40 | 16 | 49/68 | 63/84/64 | 3433 | 426 | THROTTLED, UNTHROTTLED |
| 1024 | llama-cpp-q8_0-f32-kv-p1024 | 40/41/40 | 13 | 46/64 | 50/68/52 | 3146 | 333 | THROTTLED, UNTHROTTLED |
| 1024 | llama-cpp-q8_0-f16-kv-p1024 | 40/41/40 | 7 | 46/64 | 50/69/52 | 3148 | 278 | THROTTLED, UNTHROTTLED |
| 2048 | ullm-sq8_0-f32-kv-p2048 | 40/42/40 | 12 | 57/78 | 74/96/80 | 3405 | 392 | THROTTLED, UNTHROTTLED |
| 2048 | llama-cpp-q8_0-f32-kv-p2048 | 40/41/40 | 14 | 46/62 | 55/72/58 | 3126 | 308 | THROTTLED, UNTHROTTLED |
| 2048 | llama-cpp-q8_0-f16-kv-p2048 | 40/41/40 | 16 | 47/60 | 55/72/58 | 3169 | 336 | THROTTLED, UNTHROTTLED |
| 4095 | ullm-sq8_0-f32-kv-p4095 | 40/41/40 | 13 | 69/90 | 75/96/82 | 3424 | 434 | THROTTLED, UNTHROTTLED |
| 4095 | llama-cpp-q8_0-f32-kv-p4095 | 40/41/40 | 16 | 49/66 | 65/81/70 | 3145 | 330 | THROTTLED, UNTHROTTLED |
| 4095 | llama-cpp-q8_0-f16-kv-p4095 | 40/41/40 | 7 | 50/67 | 66/81/70 | 3148 | 334 | THROTTLED, UNTHROTTLED |

`THROTTLED` and `UNTHROTTLED` are both literal AMD SMI status strings observed
in each process's stream.  Their appearance is reported as an observation
only; this record does not assign a thermal-throttle cause to a performance
result.
