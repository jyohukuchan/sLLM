# Roofline and dispatch audit

The fixed R9700 decode accounting from `sq8-r9700-handwritten-kernel-phase0-v0.1/efficiency.md` is retained exactly:

```text
B_SQ8_0(C=1036) = 15,109,299,200 logical bytes/token
measured decode = 15.294955751 tok/s
logical stream rate = 231.096063 GB/s
reference roof = 640 GB/s = 42.358020152 tok/s
eta_logical = 36.1088%
```

This is a KV-inclusive logical-stream metric, not a physical HBM counter. It cannot by itself distinguish an ALU-bound kernel from a memory-bound kernel.

More importantly, the selected-region trace for this exact full-model workload reported **zero** launches of all four generic `ullm_sq_fp8_matvec_{f32,batch,pair,triple}_kernel` symbols. It reported CK projection kernels for 40.1305% of decode kernel time and `ullm_paged_decode_attn_f32_kernel` for 50.9968%. The normal serving dispatcher selects `Sq8LayerExecutionProfile::Rdna4W8a8BlockCk`; the generic matvec is used by `ReferenceW8a16Block2d`, not this baseline serving profile.

Therefore:

1. The assertion that the 15.294955751 tok/s / 36.1088% baseline sends all projection traffic through the four generic legacy kernels is false for the captured workload.
2. No valid ALU-versus-memory roofline classification for those generic kernels can be inferred from that full-model metric. Their own physical bandwidth, occupancy, and timing were not measured in Phase 0.
3. The ISA disproves the specific recurring-division explanation for their element loop. It does not prove that the generic fallback is fast, nor does it classify the selected CK path.

Phase 1 can safely verify source-level exactness for the generic fallback, but it has no evidence-backed route to improve the current full-model baseline unless serving dispatch is changed. Dispatch changes are outside this task.
