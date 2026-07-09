# P3 CK profiler evidence

- Command: `HIP_VISIBLE_DEVICES=1 GPU_DUMP_CODE_OBJECT=1 rocprofv3 --kernel-trace --stats -f csv -d .../profiler -o ck-m8 -- build/tools/bench-sq8-ck-abscale --device 0 --m 8 --n 5120 --k 5120 --warmups 0 --repeats 1`
- Profile scope: all six CK ABScale candidates accepted for M=8, N=K=5120.
- Trace files: `profiler/ck-m8_kernel_trace.csv` and `profiler/ck-m8_kernel_stats.csv`.
- Trace SHA-256: `6de5728bc91f82b6e4a9dfc82022ef5b82c7a39203fe3bec51099ae6b21fa88e`.
- Stats SHA-256: `a78af555a7e0ea83463bafa2d9f02b3b31aa2b77c08244fc08f56d49a3eb4298`.
- gfx1201 code object 0 SHA-256: `f311c69c7c1608b5e38ea63795376dd5e41e7897fc9892241c4b3e5c69626523`.
- gfx1201 code object 1 SHA-256: `b010aec51d78a93ca5bb1c3742e6c8735022df535591bb85c22f3ceab5a537ef`.
- Both code objects contain 240 `v_wmma_f32_16x16x16_fp8_fp8` instructions according to `/opt/rocm-7.2.1/lib/llvm/bin/llvm-objdump -d --mcpu=gfx1201`.
- Example instruction: `v_wmma_f32_16x16x16_fp8_fp8 v[159:166], v[81:82], v[77:78], 0`.
- The M=8 non-profiled selection run reports kernel p50 `0.0282400008 ms`, exact all-ones output, and no fallback.

rocprofiler SDK 1.1 did not return reliable derived memory or occupancy counters for gfx1201. The trace still records the CK dispatch attributes, including a 256-thread workgroup, 34,816-byte LDS use, 224 VGPRs, and grid X=5120 for the representative selected-family dispatch. P4 records quantization-inclusive timing and actual-weight correctness separately.
