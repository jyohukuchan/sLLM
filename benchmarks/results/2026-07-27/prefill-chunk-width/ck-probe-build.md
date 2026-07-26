# Direct CK wide-M shape probe build

The probe source compiled successfully on 2026-07-27 without running a GPU
workload:

```text
/opt/rocm-7.2.1/bin/hipcc -std=c++20 -O3 \
  -DCK_USE_OCP_FP8=1 -DCK_ENABLE_FP8=1 -DCK_ENABLE_BF16=1 \
  --offload-arch=gfx1201 -I runtime/src -I /opt/rocm-7.2.1/include \
  benchmarks/results/2026-07-27/prefill-chunk-width/wide_m_ck_shape_probe.cpp \
  runtime/src/sq8_ck_gfx1201.hip.cpp \
  -L /opt/rocm-7.2.1/lib -ldevice_gemm_operations -lamdhip64 \
  -o /tmp/ullm-wide-m-ck-shape-probe
```

After the required preflight, it was run in one short controlled service
window under `flock -n /run/ullm/r9700.lock` with `HIP_VISIBLE_DEVICES=1`.
All 24 `(M, projection shape)` rows succeeded; the raw output is
[`wide-m-ck-shape-probe.jsonl`](wide-m-ck-shape-probe.jsonl). M=256 through
4096 accepted the Q/O, K/V, gate/up, and down shapes. The M=128 gate/up and
down controls selected implementation IDs 3 and 1; wide M selected IDs 2 and
4 for those two shapes, respectively.

The pre- and post-window R9700 sample was edge 37 C, hotspot 38 C, memory 36
C, and socket power 16 W, satisfying the reachable edge <=45 C gate. The
gateway was restored to `ActiveState=active`, `NRestarts=0`; the forbidden
`llama-qwen35-udq4.service` remained `inactive`/`disabled`. This is a direct
CK shape-admission check with zeroed inputs/weights, not a numerical-fidelity
or throughput benchmark.
