# SQ9_0 V620 (gfx1030) viability rerun — 2026-07-26

## Verdict

For the decode-like M=1 Qwen3-14B FP8 self_attn.q_proj shape, SQ9_0 was
shorter than SQ8_0, but did **not** clear the design gate. The best
thermal-normalized result was the cooperative-LDS SQ9_0 path at **+6.069%**
throughput relative to SQ8_0; the lane path was **+4.316%**. The required
improvement is +7.29%, using (SQ8_ms / SQ9_ms - 1) * 100, so SQ9_0 is not
viable as a decode replacement on this evidence.

There is a separate batched regime: the lane SQ9_0 path cleared +7.29% at
M=8, M=32, and the limited M=128 observation. This does not reverse the
decode decision. The exact crossover between M=1 and M=8, other shapes,
quality, and a matched model loop are unmeasured.

No runtime implementation, campaign, candidate, release, service, or
activation was changed.

## Device and safety controls

| item | value |
| --- | --- |
| selected GPU | AMD Radeon Pro V620, gcnArchName=gfx1030 |
| selected PCI BDF | 0000:03:00.0 (card0) |
| HIP visibility used for runs | HIP_VISIBLE_DEVICES=2; the isolated process therefore logs this physical card as HIP ordinal 0 |
| exact junction path | /sys/class/drm/card0/device/hwmon/hwmon5/temp2_input |
| verified label | temp2_label=junction |
| hard guard | abort at >= 85 C |
| cooldown policy | wait for <= 42 C before each warmup and timing point; record start/end/peak temperatures |

The benchmark maps hipDeviceGetPCIBusId to /sys/class/drm/card*/device before it
accepts a temp2_input; it fails closed unless that input is labelled junction.
It samples before and after every warmup and timed launch, not just between
configurations. The current source also cools after the final point and
requires both --shape and --m-values for a full suite, preventing an
accidental all-shape sweep.

The staged, usable data stayed well below the guard:

| stage / usable result | maximum junction | guard / cooldown result |
| --- | ---: | --- |
| M=1 short smoke | 42 C | completed |
| M=1 three final statistical runs | 43 C | completed |
| M=1 dequant isolation | 44 C | completed |
| M=8 two steady runs | 45 C / 46 C | completed |
| M=32 statistical run | 48 C | completed |
| M=128 limited three-trial run | 51 C | completed and cooled to 42 C |

No valid card0 run emitted thermal_guard_triggered or cooldown_timeout. Three
M=512 probes were stopped and excluded: their invocations omitted --shape, so
the old default began progressing past the intended q_proj shape. They reached
58–59 C, then waited at 43 C rather than regaining the 42 C reproducibility
target. Their exact PIDs were sent SIGKILL before the all-shape sweep could
continue further.
The card was 43 C immediately afterward, no benchmark process remained, and no
more GPU measurement was started. These partial logs are retained under
raw/stage7-* and thermal/stage7-*, but are not used below.

Historical *bdf43* logs in this directory are the earlier unsafe card1
(0000:43:00.0) attempt and are preserved only as history. They are not part of
this rerun's result.

## Fidelity checks and layout

All four GPU paths (SQ8_0, both SQ9_0 high-plane variants, and FP16) passed
the CPU-reference GEMV and dequant checks before timing. The largest recorded
absolute error was 1.192092896e-7 for SQ8_0; every other recorded maximum was
zero.

The tested shape is Qwen3-14B FP8 self_attn.q_proj, 5,120 by 5,120. The SQ8_0
layout was reconstructed from the in-tree implementation: F8 E4M3 payload plus
a row-major 128x128 BF16 scale grid, expanded to F32 scales for the V620
fallback resident representation. The exact resident sizes are:

| representation | bytes |
| --- | ---: |
| SQ8_0 F8 payload | 26,214,400 |
| SQ8_0 artifact BF16 scales | 3,200 |
| SQ8_0 artifact total | 26,217,600 |
| SQ8_0 V620 resident F32-scale total | 26,220,800 |
| SQ9_0 low plane | 26,214,400 |
| SQ9_0 high/sign plane | 3,276,800 |
| SQ9_0 total | 29,491,200 |

Thus SQ9_0 is 1.12472540884 times the actual fallback resident SQ8_0
representation, a **12.4725408836%** increase rather than an assumed 12.5%.
The SQ9_0 tile is the specified aligned 128-byte low plane plus 16-byte high
plane (144 bytes total) and reconstructs the full 9-bit code with one q << 7
shift.

## M=1 (decode-like) comparison

These are medians of the three per-run medians from
raw/final-m1-r{1,2,3}-card0-v4.jsonl. Each run used 32 warmups and 31 timed
single-launch trials; timing starts were 41–42 C, ends 42–43 C, and every peak
was 43 C. One lane run showed an AUTO-DPM acceleration outlier (0.515765 ms);
using the median across the three independent run medians intentionally rejects
it.

The bandwidth column is the benchmark's modeled weight stream, and the
efficiency uses its declared 512 GB/s reference. It is not a profiler-derived
physical-memory transaction count or an independently revalidated V620 peak.

| format / implementation | median ms | modeled GB/s | % of 512 GB/s | ns / logical FMA | throughput vs SQ8_0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SQ8_0 F8 E4M3 + F32 block scale fallback | 0.639007 | 41.034 | 8.014% | 0.0243762 | baseline |
| SQ9_0 per-lane high byte | 0.612567 | 48.144 | 9.403% | 0.0233676 | **+4.316%** |
| SQ9_0 cooperative LDS high plane | 0.602446 | 48.952 | 9.561% | 0.0229815 | **+6.069%** |
| FP16 raw reference | 0.589446 | 88.946 | 17.372% | 0.0224856 | +8.408% vs SQ8_0 |

Both SQ9_0 variants are faster in raw M=1 elapsed time, but neither reaches
the required +7.29%. The SQ8_0 M=1 path is 8.408% slower than the raw FP16
reference in this same synthetic shape, which is consistent with exposed
fallback work, not a proof of a pure bandwidth limit.

## M sweep

All rows use the same q_proj shape and 42 C cooldown target. A positive value
is the lane SQ9_0 throughput improvement (SQ8_ms / SQ9_ms - 1) * 100.

| M | SQ8_0 ms | lane SQ9_0 ms | improvement | evidence / thermal peak | interpretation |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.639007 | 0.612567 | +4.316% | three 31-trial runs, 43 C | fails gate |
| 8 | 1.241054 | 0.989531 | +25.418% | mean of two 15-trial run medians, 45/46 C | clears gate in a batched regime |
| 32 | 5.861702 | 4.631769 | +26.554% | one 9-trial run, 48 C | clears gate; one run only |
| 128 | 24.630720 | 19.803831 | +24.374% | one limited 3-trial run, 51 C | clears gate; limited statistics |
| 512 | unmeasured | unmeasured | — | partial probes excluded at 58–59 C | no statistical conclusion |

The first observed regime change is therefore somewhere between M=1 and M=8;
M=2–7 were not measured. M=8 and M=32 samples include initial AUTO-DPM ramp
outliers (for example, 6.2–6.4 ms first samples), so their reported medians are
useful conditional evidence, not a fixed-clock performance claim.

## Isolated dequant evidence and ISA

The guarded M=1 dequant isolation run recorded:

| path | dequant-only ms | raw-payload control ms | comparison |
| --- | ---: | ---: | --- |
| SQ8_0 fallback | 0.245603 | 0.110122 | dequant path is 123.03% longer |
| SQ9_0 lane | 0.195802 | 0.274683 | control is not comparable as a load-only baseline |
| SQ9_0 LDS | 0.257602 | 0.164961 | dequant path is 56.16% longer |
| FP16 | 0.141682 | 0.147961 | near the control |

The raw-payload control deliberately records
raw_payload_stream_control_is_load_only=false; it is therefore evidence for
the cost of this kernel construction, not a stand-alone memory-bandwidth
measurement. Still, SQ8_0 has a clear 0.135481 ms exposed dequant/control cost
in this comparison. The offline disassembly corroborates that the SQ8_0
dequant kernel has 377 static v_* instructions, versus 250 for the lane SQ9_0
implementation, and carries its E4M3 reconstruction and block scale path.
Full resources, symbols, and reproducible commands are in
[static/isa-analysis.md](static/isa-analysis.md).

This supports the statement that SQ8_0 fallback dequant has a material
ALU/control component. It does not prove that every full M=1 GEMV is purely
ALU-bound; the M=1 gate remains the decision-making measurement.

## Files and reproducibility

- Source: tools/bench-sq9-v620-viability-hip.cpp
- Build helper: tools/build-bench-sq9-v620-viability.sh
- Final safety-only rebuild: build/bench-sq9-v620-viability-hip-thermal-v6
  (SHA-256 7ebe8e096b385d0dbfb0bacf7bdc577162ff755ed503c0d5fa5a9b57b3799e95)
- Valid timing raw data: raw/final-m1-r*-card0-v4.jsonl,
  raw/final-m8-steady*-card0-v4.jsonl,
  raw/final-m32-stats-card0-v4.jsonl, and
  raw/final-m128-limited-card0-v5.jsonl
- Corresponding temperature histories: thermal/ with the same stems
- Preflight mapping: raw/preflight-card0-visible2-v4.jsonl
- Dequant isolation: raw/stage3-m1-dequant-card0-visible2-v1.jsonl

Build without touching any production tree:

~~~bash
tools/build-bench-sq9-v620-viability.sh \
  benchmarks/results/2026-07-26/sq9-v620-viability/build/bench-sq9-v620-viability-hip-thermal-v6
~~~

Future full-suite invocations must explicitly name both a shape and the M list,
for example --shape qwen3_14b_q_proj --m-values 1; omitting either is now a
fail-closed error. No further V620 execution was performed after the M=512
thermal stop.
