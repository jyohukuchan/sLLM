# gfx1030 ISA/resource comparison

Both sides were compiled device-only from the runtime HIPRTC source with the
same ROCm 7.2.1 toolchain. The reference evidence is the committed
`../../sq8_1/static-reference-kernel/` artifact (source SHA-256
`8815118714bbec10b441238088e59828b496b26ba746afeaee29f7694a9ea2df`);
this directory is the optimized source (SHA-256
`88489aa38dbddd297f59f91f27a22bbfa81e7b5677d6d64282ed78af75ce15a3`).

| gfx1030 kernel | reference | optimized | evidence/meaning |
| --- | ---: | ---: | --- |
| W8A16 fixed LDS | 1,024 B | 0 B | wave32 shuffle reduction removes the tree buffer |
| W8A16 `s_barrier` static count | 2 | 0 | no workgroup synchronization remains |
| W8A16 VGPR / SGPR | 55 / 28 | 58 / 22 | no spills on either side; the small VGPR increase bought barrier removal |
| W8A16 `global_load_dwordx4` static count | 10 | 10 | aligned vector payload-load form is retained; scalar payload access is tail-only |
| W8A8 fixed LDS | 1,024 B | 0 B | optimized kernel uses dynamic LDS instead of fixed LDS |
| W8A8 dynamic LDS at K=5120 | 0 B | 5,760 B | 5,120 activation codes + 160 F32 activation scales, shared by eight rows |
| W8A8 `s_barrier` static count | 2 | 1 | one barrier publishes the shared activation plane |
| W8A8 VGPR / SGPR | 53 / 59 | 39 / 32 | zero private bytes and zero VGPR/SGPR spills on both sides |
| W8A8 `v_dot4c_i32_i8` static count | 16 | 8 | optimized full-block body has eight explicit dot4 operations; see dynamic accounting below |
| W8A8 divide sequence (`v_div_*` + `v_rcp`) | 165 | 10 | static code-body reduction from quantization restructuring; includes control/tail code |
| W8A8 `v_rndne_f32` static count | 32 | 1 | static code-body reduction; dynamic amortization is the meaningful metric |

The analyzer JSONs are the authority for the counts:
`sq8_1-reference-w8a16-gfx1030.json` and
`sq8_1-reference-w8a8-gfx1030.json`. Static counts cover all emitted paths
(including tail/control) and must not be treated as an exact instruction count
per element.

## Per-complete-K=32 dynamic accounting

- Payload: W8A16 and W8A8 both issue two 16-byte `uint4` payload loads for 32
  int8 codes. That is 32 B / 32 elements and two 128-bit loads / 32 elements
  = 1/16 vector-load instructions per element. The gfx1030 disassembly retains
  `global_load_dwordx4`; the guarded scalar load is tail-only.
- W8A8 dot: each output row needs eight signed dot4 operations for 32 values:
  8 / 32 = 0.25 dot4 instructions per element both before and after. The
  optimized code does not claim a false reduction in required integer MACs.
- W8A8 activation quantization: the reference performs the 32 value
  scale/divide/round operations independently for every output row. The tiled
  kernel performs them once for an eight-row tile: 32 / 8 = four amortized
  activation quantizations per output row/K=32, an exact 8× reduction in this
  work. Its two shared `ds_read_b128` loads recover that common activation
  plane for a full K=32 block.

The 8-row/256-thread tile was selected from these static resource values, not
from an assumed occupancy value: it is spill-free, uses 39 VGPR / 32 SGPR for
the W8A8 hot path, and its measured-shape dynamic LDS request is 5,760 B under
the 48 KiB runtime cap. Achieved occupancy was not collected with a profiler,
so a numerical occupancy percentage is intentionally unreported.
