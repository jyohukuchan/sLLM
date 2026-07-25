# gfx1030 dequant ISA analysis

This is an offline analysis only; it did not execute a GPU kernel.  It covers
the `dequant_sum_kernel` specializations embedded in
`build/bench-sq9-v620-viability-hip-thermal-v5` (whole executable SHA-256
`5cb7e192e3fc3c668fea88afb6121bc7ef04d6c74b8db5d606ef0203e56c5632`).
The gfx1030 code object is at offset 36,864 with size 218,704 bytes.  v6 adds
only the host-side fail-closed `--shape` requirement; these kernel bodies are
unchanged.

## Reproduction

```bash
bin=benchmarks/results/2026-07-26/sq9-v620-viability/build/bench-sq9-v620-viability-hip-thermal-v5
roc-obj-ls -v "$bin"
dd if="$bin" bs=1 skip=36864 count=218704 status=none |
  /opt/rocm-7.2.1/lib/llvm/bin/llvm-readobj --notes -
dd if="$bin" bs=1 skip=36864 count=218704 status=none |
  /opt/rocm-7.2.1/lib/llvm/bin/llvm-objdump --mcpu=gfx1030 \
  --disassemble-symbols='<mangled dequant_sum_kernel symbol>' -
```

## Compiler resources

| format / specialization | code bytes | VGPR | SGPR | LDS bytes | private bytes / spills | wave |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `SQ8_0`, `I0,B0` | 4,232 | 17 | 22 | 1,152 | 0 / 0 | 32 |
| `SQ9_0` lane high-byte, `I1,B0` | 2,796 | 28 | 26 | 1,024 | 0 / 0 | 32 |
| `SQ9_0` cooperative LDS high-plane, `I1,B1` | 2,512 | 18 | 23 | 1,536 | 0 / 0 | 32 |
| FP16 reference, `I3,B0` | 2,036 | 18 | 18 | 1,024 | 0 / 0 | 32 |

The symbol names use the template arguments above, for example
`_ZN12_GLOBAL__N_118dequant_sum_kernelILi0ELb0EEEvPKhPKfS2_PKtPfmmmm` for
`SQ8_0` and `...ILi1ELb0...` for the lane `SQ9_0` path.

## Static instruction evidence

The compiler unrolls 16 logical values in the hot path.  The following counts
are counts in the emitted static body, not dynamic instructions per logical
element; they include address generation, reductions, and control, so they
must not be divided by 16 and labelled as a precise VALU-per-element number.

| specialization | all `v_*` instructions | `v_lshlrev_b16` | `v_cvt_f32_f16` | `global_load_ubyte` | `global_load_dwordx4` | `ds_read_b32` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SQ8_0` | 377 | 0 | 0 | 0 | 1 | 17 |
| `SQ9_0` lane high-byte | 250 | 16 | 16 | 16 | 1 | 1 |
| `SQ9_0` LDS high-plane | 218 | 16 | 16 | 0 | 2 | 17 |
| FP16 reference | 191 | 0 | 16 | 0 | 2 | 1 |

For each fully active 16-value unroll, both `SQ9_0` paths therefore contain
exactly one `v_lshlrev_b16` and one `v_cvt_f32_f16` for each value: this is the
specified whole-code `q << 7` reconstruction followed by the half-to-float
conversion.  The lane form pays 16 byte high-plane loads, while the LDS form
trades those for shared-memory traffic.  The FP16 reference has the 16
half-to-float conversions but no reconstruction shifts.

`SQ8_0` has the largest static vector instruction body (377 versus 250 for
the lane `SQ9_0` form) and also performs the E4M3 reconstruction plus the
block-scale path.  This is evidence of a substantial conversion/control
burden, but it is not by itself a proof that a full GEMV is at the theoretical
ALU roof.  The corresponding guarded dequant-only measurements are recorded
in `../raw/stage3-m1-dequant-card0-visible2-v1.jsonl` and must be interpreted
together with their non-load-only control.
