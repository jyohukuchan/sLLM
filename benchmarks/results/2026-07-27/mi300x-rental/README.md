# MI300X rental P0 and hardware microbenchmark evidence

Remote checkout: `c5e7dc16c702e8bdada7da001ee8bc15f728b088` plus the
rental-runner, hardware-benchmark, ISA-audit, and physical-smoke files synced
from this worktree. No model, GGUF, Docker image, or external engine was
downloaded.

Provisioning began at `2026-07-27T04:52:27+00:00`. Rust minimal, checkout, and
`cargo fetch --locked` completed by `04:53:42+00:00` (about 75 s); P0 began at
`04:54:04+00:00`. The fetch log records 29 downloaded archives. `environment.txt`
records the rental linker override: `cc` and cleared Rust flags.

P0 timing is in `stage-timings.tsv`: preflight 0 s, CPU 82 s, HIPRTC 32 s,
build 54 s, ISA 4 s, physical 2 s. `logs/physical.log` contains all five A′
shapes with zero error. `logs/b-control-physical-recheck.log` is the explicit
hardware B-control recheck and reports `first=0.53125 expected=0.53125`.
The original CPU-stage focused test placed Cargo global flags after `--`, so
`logs/b-control-cpu-recheck.log` is the authoritative successful focused
recheck after the runner fix.

`hw-microbench/` contains build/ISA records, numeric validation, HIP-event
throughputs, runtime phase timing, and continuous amd-smi telemetry. The final
successful optional stage took 34 s. Earlier optional-stage records document
two compatibility discoveries before measurement: the checkout lacked the ISA
helper, and amd-smi 26.2.2 removed `--violation`; neither started a benchmark
measurement. The runner now samples the portable temperature/clock/power set.
