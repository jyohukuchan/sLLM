# SQ8_1 dot4 architecture recheck — 2026-07-26

## Scope

This is an offline compiler check only.  It does not call a HIP runtime API,
enumerate a GPU, select a device, allocate device memory, or launch a kernel.
In particular, R9700/gfx1201 was not used.

## Source and compiler

- Probe source: `tools/sq9-q8-gfx1030-isa.hip.cpp`
  (`SHA-256 908774b2a08b14bdbba702b9b0f1bd446260cf048756bdd34e71311cc9fe8b70`).
- Compiler: HIP `7.2.53211-e1a6bc5663`, AMD clang `22.0.0git`, ROCm `7.2.1`.
- Command shape: `/usr/bin/hipcc -O3 -std=c++17 --offload-arch=<gfx> --offload-device-only
  tools/sq9-q8-gfx1030-isa.hip.cpp -o <temporary bundle>`.
- The named probe is the fixed-K=128 `ullm_q8_0_w8a8_g32_gemv_isa` kernel and calls
  `__builtin_amdgcn_sdot4`.

## Result

| Target | Compiler result | Static W8A8 result |
| --- | --- | --- |
| gfx1030 | success | 32 `v_dot4c_i32_i8` instructions in the named K=128 symbol |
| gfx942 | success | 32 `v_dot4c_i32_i8_e32` instructions in the named K=128 symbol |
| gfx1201 | compile failure | `__builtin_amdgcn_sdot4` needs target feature `dot1-insts` |

The gfx1201 compiler diagnostic was:

```text
tools/sq9-q8-gfx1030-isa.hip.cpp:25:12: error: '__builtin_amdgcn_sdot4' needs target feature dot1-insts
```

The generated temporary artifacts were retained only for this inspection. Their SHA-256 values were
gfx1030 disassembly `23d8aa4551c8d38abde14c3fb5590ea386cef9429b7a97bb05f5571222c82237`, gfx942
disassembly `d0575bc5d0169b2c68f8166b5c6e64a563fda15434d0a565202f2728591cf8f8`, and gfx1201
compiler stderr `4bb6e1b9511341275db5b308a664e7bc193c5609cb355e64b03902d05c0ff353`.

This establishes compiler/ISA eligibility, not GPU performance.  It supports `SQ8_1` W8A8
candidate dispatch on gfx1030 and gfx942 only; gfx1201 remains on `SQ8_0` in this design.
