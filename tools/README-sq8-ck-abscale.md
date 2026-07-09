# SQ8 Composable Kernel ABScale probe

This standalone probe checks the ROCm 7.2.1 prebuilt Composable Kernel ABScale instances for
FP8 E4M3 OCP `A[M,K] * B[K,N] -> BF16 C[M,N]` with scale blocks `1x128x128`.
It does not modify or select the production uLLM runtime path.

CK sees B as column-major `[K,N]` with leading dimension K. Therefore its physical byte offset is
`n*K + k`, which is exactly the existing canonical SQ8 weight layout stored row-major as `[N,K]`.
No weight transpose or repack is performed by the probe.

Build and run on the R9700 (physical HIP device 1 on the current host):

```bash
tools/run-sq8-ck-abscale.sh
```

When neither `--device` nor `HIP_VISIBLE_DEVICES` is supplied, the script uses
`ULLM_CK_ABSCALE_DEVICE_SELECTOR=1` and runs the isolated device as ordinal zero. An explicit
`--device 1` is also accepted and produces the same isolation automatically.

On a multi-GPU host, the probe always runs with the requested device as the only
`HIP_VISIBLE_DEVICES` entry before initializing HIP. This is required because the prebuilt CK fat
binary otherwise binds its kernel registration to the process's default gfx1030 device. If the
caller already supplied `HIP_VISIBLE_DEVICES`, `--device` is resolved as an ordinal within that
list and its token is isolated, so the caller's mapping is preserved. The JSON records the original
visibility list, requested ordinal, selected token, and remapped internal device zero.

The defaults are `M=8`, `N=5120`, `K=5120`, five warmups, and twenty measured repeats. Override
them with `--m`, `--n`, `--k`, `--warmups`, and `--repeats`. The script always compiles with
`-DCK_USE_OCP_FP8=1`; `--build-only` compiles without running the probe.

Successful stdout is one JSON document containing all four CK instance groups, support counts,
the fastest correct instance, an all-ones maximum absolute difference, and HIP-event kernel p50.
The probe never falls back. A device other than gfx1201 or a shape with no supported instance exits
nonzero and emits an error JSON document.
