# Gemma4 resident host-gap localisation v0.1

## Decision

The Gemma4 decode gap is primarily activation **D2H round-trip submission
latency**, not allocation and not bare kernel launch overhead.

The resident executor keeps source BF16 weights and K/V on the R9700, but it
still uses the host as the activation graph: every BF16 projection/row read
and attention result is copied back to a freshly allocated host byte vector,
the stream is synchronized, the result is decoded to F32 and transformed on
the CPU, then the next input is copied to the device. Four decode tokens make
1,108 matvec, 1,056 BF16-row, 140 attention, and 60 K/V-write calls: 2,304
host-visible result round trips.

## Reproducer

The service was stopped and `/run/ullm/r9700.lock` was acquired with `flock`.
Only `HIP_VISIBLE_DEVICES=1` (the R9700) was exposed:

```text
flock -n /run/ullm/r9700.lock env \
  HIP_VISIBLE_DEVICES=1 \
  ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 \
  ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 \
  ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 \
  rocprofv3 --kernel-trace --runtime-trace --output-directory \
  benchmarks/results/2026-07-27/gemma4-host-gap-localization-v0.1/raw/gemma-benchmark-rocprof \
  -- target/release/ullm-gemma4-resident \
  --model-dir /home/homelab1/datapool/ai_models/safetensors/gemma-4-E2B \
  --output benchmarks/results/2026-07-27/gemma4-host-gap-localization-v0.1/raw/gemma-benchmark-host-profile-rocprof.json \
  --mode benchmark --benchmark-repeats 3
```

`rocprofv3` is deliberately retained here to match DD's harness. Its own
instrumentation lowers the observed decode rate to 11.909 tok/s, so these are
localisation timings, not the throughput baseline. The unprofiled companion
run is also retained in `raw/gemma-benchmark-host-profile.json` and measured
15.036 tok/s.

## Decode host profile

The table aggregates DD's same 3 x 4-token decode shape. Values are exclusive
within `primitive_ns`, except that `stream synchronize` is a wait for work
already enqueued by the preceding copies/kernels; it must therefore not be
added to an independent GPU-kernel duration. The final residual is primitive
validation/bookkeeping not split further.

| contributor | ms / 12 decode tokens | share of 1,007.606 ms measured wall | interpretation |
| --- | ---: | ---: | --- |
| D2H submission call | 473.378 | 46.98% | `copy_to_host` itself blocks repeatedly while moving every activation result into pageable host storage |
| `hipStreamSynchronize` wait | 222.711 | 22.10% | completion of the preceding result copy and queued GPU work before CPU consumes it |
| CPU executor work outside primitives | 167.311 | 16.61% | RMSNorm, RoPE, GELU, residuals, PLE, finite checks, registry/descriptor work between primitive calls |
| kernel submission / wrapper validation | 62.069 | 6.16% | argument/range validation plus the primitive runtime call; not bare HIP launch alone |
| H2D submission call | 43.455 | 4.31% | serializing and uploading the next activation/K/V/query |
| decode F32 and validate | 20.014 | 1.99% | bytes-to-F32 conversion plus finite checks |
| F32 host encode | 9.679 | 0.96% | F32-to-bytes conversion before H2D |
| output `Vec<u8>` allocation | 4.598 | 0.46% | one transient output vector per host readback |
| K/V page-table host work | 1.686 | 0.17% | sliding-cache table preparation and submission |
| buffer checks | 0.443 | 0.04% | retained transient-buffer size checks |
| unsplit primitive bookkeeping | 2.262 | 0.22% | inclusive primitive time less the named primitive subcategories |
| **accounted total** | **1,007.606** | **100.00%** | sum of the non-overlapping measured categories |

There were **zero** transient device-buffer allocations in all timed decode
runs. This refutes allocation as an explanation for DD's 746.578 ms
outside-kernel total. D2H call latency plus the host activation graph alone
accounts for the bulk; it is fully consistent with DD's finding that the next
kernel launch had generally not returned yet: the executor had not reached
that next launch because it was still returning and processing the prior
activation.

## Instrumentation contract

`Gemma4ResidentHostProfile` is emitted in each resident benchmark run. It
measures encode, buffer checks/allocation, H2D/D2H submission, primitive
submission, synchronization, F32 decode/validation, K/V table work, and the
remaining CPU executor interval. It does not change numerical operations or
runtime source; `cargo check -p ullm-engine --bin ullm-gemma4-resident`
completed successfully.
