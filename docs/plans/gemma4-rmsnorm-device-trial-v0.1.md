# Gemma4 direct-weight RMSNorm device trial v0.1

## Verdict: rejected and rolled back

The first candidate port used the existing direct-weight `ullm_rmsnorm_f32`
kernel. It converted the resident BF16 gamma on device, ran RMSNorm on device,
then returned the normalized activation to the still-host-resident consumer.
This removed the individual norm-weight row readback but did **not** retain the
activation edge on device. The extra H2D, kernel launch, and output D2H made
the complete decode slower, so the executor wiring was removed before commit.

## Real-activation validation

`ULLM_GEMMA4_VALIDATE_DEVICE_RMSNORM=1` compared the device result to the
unchanged Rust direct-weight reference for 44,202 RMSNorm calls / 32,238,336
real activation elements across the full resident validation suite:

| max abs | max rel | full-model result |
| ---: | ---: | --- |
| 0.000732421875 | 0.000002284354877701844 | both four-step continuations match the HF-trace IDs; cache/full-reprefill checks pass |

The finite difference is expected from the existing kernel's parallel
reduction and `rsqrtf` versus the reference's serial accumulation and `powf`.
It was not accepted as a standalone graph port merely because text remained
coherent: the performance result fails the objective.

## Measurement

R9700-only, three-repeat harness result after the temporary port:

| path | decode tok/s | prefill tok/s |
| --- | ---: | ---: |
| per-primitive baseline | 15.733 | 18.544 |
| temporary RMSNorm edge | **12.297** | **13.078** |

Raw evidence is retained under
`benchmarks/results/2026-07-27/gemma4-activation-device-v0.1/raw/` as
`rmsnorm-real-activation-validation.json` and `rmsnorm-port-benchmark.json`.

The rollback is source-clean against the preceding ranking commit. The next
attempt must first retain the producing matvec output and the RMSNorm output in
persistent device workspaces; a host-visible RMSNorm output is not viable.
