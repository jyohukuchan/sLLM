# MI300X rental CPU benchmark — 2026-07-27

## Scope and limitations

This is a measurement of the **13-vCPU KVM guest**, not a measurement of the
whole Xeon socket.  The rental configuration says that these vCPUs are a slice
of a 52-core Intel Xeon Platinum 8470 (Sapphire Rapids).  Consequently none of
the bandwidth values below may be described as the socket's DDR5 peak.  They
are the bandwidth visible from this VM.  The guest is shared infrastructure,
so other tenants can affect it; raw per-repeat timing and load-average samples
are retained here.

The guest exposes one NUMA node (`0`, CPUs `0-12`) and 13 vCPUs, one thread per
guest core, one guest socket.  Its visible cache hierarchy is L1d 416 KiB, L1i
416 KiB, L2 52 MiB, and L3 16 MiB.  `lscpu -e` and `/proc/cpuinfo` report 2.0
GHz for all visible CPUs; guest min/max MHz are not exposed.

`numactl --cpunodebind=0 --membind=0` succeeded for the adopted measurements.
The timing configuration used an independent 512 MiB vector per STREAM array:
32 times the exposed 16 MiB L3, so it cannot reside in that cache.  Each metric
has 3 warmups and 7 timed repetitions; the reported value is the median.
Allocation, initialization, warmups, checksum consumption, and JSON output are
outside each timed interval.  Read/copy/triad use conventional STREAM traffic
accounting of 1/2/3 times the vector size, respectively (copy's potential
write-allocate traffic is not included in that convention).

## CPU ISA actually exposed to the guest

Source: the captured first `flags` line of `/proc/cpuinfo` and `lscpu` in
[`raw/mi300x-cpu-bench-20260727/cpu-topology.txt`](raw/mi300x-cpu-bench-20260727/cpu-topology.txt).

- AVX-512: `avx512f`, `avx512dq`, `avx512ifma`, `avx512cd`, `avx512bw`,
  `avx512vl`, `avx512vbmi`, `avx512vbmi2`, `avx512vnni`, `avx512bitalg`,
  `avx512vpopcntdq`, `avx512_bf16`, and `avx512_fp16`.
- Related non-AVX-512 flag: `avx_vnni`.
- AMX: **not exposed**.  None of `amx_tile`, `amx_bf16`, or `amx_int8` occurs
  in the guest flags.  Therefore no AMX instructions were executed or claimed.
  The processor model name alone is not treated as proof that AMX is available
  to this VM.

## NUMA, DIMM/channel information, and theoretical DDR5 bandwidth

`numactl --hardware` shows one node with 225484 MB, and cgroup effective CPU
and memory-node masks are `0-12` and `0`.  `sudo dmidecode -t memory` reports
14 virtual 16-GB QEMU DIMM devices, but every device has `Data Width: Unknown`,
`Speed: Unknown`, and `Configured Memory Speed: Unknown`.  It contains no
memory-channel mapping.  Thus the physical channel count and DDR5 transfer
rate are **unconfirmed**, and no theoretical DDR5 bandwidth is calculated.
In particular, 14 presented virtual DIMMs are not treated as 14 channels.

## Measured VM memory bandwidth

All values are GB/s (10^9 bytes/s) using the STREAM accounting stated above.
`sample seconds` is the min--max among the seven timed repetitions and makes
the observed variation explicit.

| Threads | Read | Copy | Triad |
| ---: | ---: | ---: | ---: |
| 1 | 5.650 (0.094972--0.095054 s) | 12.396 (0.086535--0.086680 s) | 13.590 (0.118435--0.122157 s) |
| 4 | 22.377 (0.023974--0.024021 s) | 28.609 (0.037508--0.037584 s) | 33.377 (0.048177--0.049335 s) |
| 8 | 42.685 (0.012567--0.014059 s) | 33.788 (0.031719--0.031840 s) | 38.824 (0.041456--0.042221 s) |
| 13 | 56.351 (0.009448--0.011887 s) | 42.922 (0.024805--0.027398 s) | 49.347 (0.032628--0.032744 s) |

The preflight host load average was `0.00 / 0.15 / 0.31`; the process listing
had no competing CPU-heavy workload.  Per-sample 1-minute load average ranged
from 1.63 to 1.75 during STREAM.  This partly reflects the benchmark itself;
it is retained rather than assumed to mean a silent machine.  See the raw JSONL
for every before/after load sample.

## AVX-512 arithmetic throughput

The separate CPU microbenchmark follows the GPU benchmark's useful convention:
warmup then median timing of the kernel-only region, with achieved TFLOPS
reported.  FP32 uses eight independent 512-bit FMA accumulators (256 FLOP per
loop iteration); BF16 uses eight `_mm512_dpbf16_ps` accumulators (512 FLOP per
iteration because each FP32 lane consumes two BF16 products).  Both use 2
warmups and 5 timed repetitions, and no peak ratio is asserted.

| Threads | AVX-512 FP32 FMA TFLOPS | AVX-512 BF16 dot-product TFLOPS |
| ---: | ---: | ---: |
| 1 | 0.190 | 0.190 |
| 4 | 0.759 | 0.759 |
| 8 | 1.518 | 1.517 |
| 13 | 2.464 | 2.439 |

The per-repeat elapsed-time ranges were 0.269873--0.338182 s (FP32) and
0.539672--0.564195 s (BF16), depending on thread count; full samples include
load values (1.23--2.86) in `compute.jsonl`.  AMX BF16/INT8 throughput is
unavailable because the necessary AMX CPUID flags were not exposed.

## SGLang CPU inference

SGLang did **not** run, and no model was downloaded.  This is an environment
provisioning failure rather than an inference result:

1. `sglang`, `torch`, `transformers`, and `llama_cpp` were all absent;
   `sglang serve --help` exited 127 (`command not found`).
2. System `pip install --dry-run sglang` exited 1 due to Ubuntu's PEP 668
   externally-managed Python environment.
3. Creating an isolated venv also exited 1 because `ensurepip` / `python3.12-venv`
   was absent.

The exact commands and diagnostics are in `sglang-availability.txt`,
`sglang-dry-run.txt`, and `sglang-venv-probe.txt`.  Installing OS packages plus
SGLang/PyTorch and downloading a 1--3B model would materially extend the
rental window, so the optional llama.cpp fallback was not started after this
conclusive SGLang provisioning probe.  Accordingly there are no prefill,
decode, or tok/s figures, and no substitute result is mislabeled as SGLang.

## Artifacts and timing

- CPU source: [`tools/cpu-microbench-sapphirerapids.cpp`](../../../../tools/cpu-microbench-sapphirerapids.cpp)
- Exact remote binary and source/binary SHA-256: [`bin/`](bin/) and [`sha256sum.txt`](sha256sum.txt)
- Raw remote capture (including architecture, stderr, JSONL, and pre/post
  process lists): [`raw/mi300x-cpu-bench-20260727/`](raw/mi300x-cpu-bench-20260727/)
  and its transport archive [`raw/mi300x-cpu-bench-20260727.tar.gz`](raw/mi300x-cpu-bench-20260727.tar.gz).
  The archive preserves the remote capture byte-for-byte; the two expanded
  text copies with terminal trailing whitespace were normalized solely so the
  repository's whitespace check remains clean.

Remote UTC wall-clock stages: read-only contention/configuration check began
05:29; CPU topology capture/build began 05:32:45; a successful locality
preflight STREAM run took 05:33:24--05:33:38 (14 s); the adopted STREAM run
took 05:33:38--05:33:53 (15 s); AVX-512 compute took 05:35:39--05:36:03 (24
s); SGLang availability/provisioning probes ran 05:36--05:38.  The gap includes
copying the revised compute binary and incremental local artifact capture.  The
measurement host was quiet at the preflight snapshot.  The prescribed
CR-marker check initially returned one match, so GPU/CPU-heavy activity was
also checked directly on the rental host before proceeding; no such activity
appeared in either process snapshot.  This CPU work did not start or stop CR.
