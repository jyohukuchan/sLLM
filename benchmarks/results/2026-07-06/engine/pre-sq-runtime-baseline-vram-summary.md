# Pre-SQ Materialized-AQ Baseline VRAM Summary 2026-07-06

## Completed Runs

| case | status | verified | prefill tok/s | decode tok/s | total wall s | consumed GiB | peak total GiB | KV bytes |
| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ullm-qwen35-9b-p4p46-baseline-r9700-pp512-tg256-vram` | ok | true | 2.912 | 0.140 | 1998.935 | 26.257 | 26.424 | 50331648 |

## Intentionally Stopped Runs

| case | reason | partial evidence |
| --- | --- | --- |
| `ullm-qwen35-9b-p4p46-baseline-v620-pp512-tg256-vram` | R9700 baseline and accepted-package R9700/V620 runs already showed stable decode around `0.14 tok/s`; completing another 256-token decode would add little information. | 294 memory samples; `card1` reached 26.268 GiB used and total used reached 26.411 GiB. |

## Interpretation

The current materialized-AQ baseline is useful as a lower-bound anchor, but repeating long decode runs on this path is not useful until the runtime path or sq candidate changes. Future runs should separate long prefill pressure from short decode probes.
