# SQ8_0 gfx1030 static analysis

## Input and non-interference gate

The compiler input was the exact HIPRTC literal extracted from
`runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc`, not a hand-copied kernel.
`sq8_0_matvec_hiprtc_static_row256.hip.cpp` and its final re-extraction compare
identically (`cmp=0`); their SHA-256 is
`d1432840ef70ccb70d5484df415ea89e40f90583b0e298bf8608aa0d02463b38`.

The source-level gate hashes the legacy `#else` bodies used by non-gfx1030
targets. It passed with these byte-stable hashes:

| body | SHA-256 |
| --- | --- |
| single matvec | `7b10c5d38ba6cc79ce346d81f9a2382bbdb8cfe5adac2b5504c6c799ff66368a` |
| batch matvec | `4aee9d87c84d6f744469c46fb896240da057a189b49b1c2227aba5c57d16ccdb` |

The extracted source was compiled device-only with ROCm 7.2.1 / AMD clang 22
for `gfx1030` and `gfx1201`. The normalized `gfx1201` baseline and final
disassemblies compare identically (`cmp=0`), and the corresponding metadata
comparison also returned `cmp=0`. Thus the gfx1201 generic bodies and code
object are unchanged by this specialization. No gfx1201 device execution was
performed.

## gfx1030 resource result

| kernel | version | VGPR | SGPR | fixed LDS | max WG | private / spills |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| direct | legacy generic | 22 | 42 | 1024 B | 1024 | 0 / 0 |
| direct | final gfx1030 branch | 31 | 48 | 32 B | 256 | 0 / 0 |
| batch | legacy generic | 22 | 47 | 1024 B | 1024 | 0 / 0 |
| batch | final gfx1030 branch | 31 | 52 | 32 B | 256 | 0 / 0 |

The host already launches 256 threads for this route; changing the metadata
maximum to 256 does not change its public launch ABI or dispatch. The relevant
trade-off is an extra 9 VGPR / 6 SGPR on the direct kernel in exchange for a
32x fixed-LDS reduction and no spills.

The isolated normal and `__launch_bounds__(256, 2)` prototypes each compiled
to 30 VGPR / 48 SGPR / 32 B LDS with zero spills. Since the explicit two-CTA
constraint gave no resource-class improvement, the final runtime source uses
only `__launch_bounds__(256)`.

## Direct-kernel ISA count

Counts are within the direct symbol body in the final/baseline gfx1030
disassemblies; they are not a hardware counter measurement.

| instruction family | legacy | final | interpretation |
| --- | ---: | ---: | --- |
| `global_load_dwordx4` | 0 | 1 | aligned `uint4` payload load admitted |
| `ds_bpermute` | 0 | 5 | five shuffle-reduction steps |
| `s_barrier` | 2 | 1 | LDS tree barrier removed; one cross-wave handoff remains |
| `ds_read` | 3 | 2 | fewer LDS reads |
| `ds_write` | 2 | 1 | one write per wave partial |

The final source takes the wide path only when the 16-byte segment is aligned
and lies wholly within a compatible scale segment. The scalar path remains for
scale boundaries, misalignment, and tails, preserving the generic semantic
contract.

## Limit

Static metadata and disassembly establish compiler resources and instruction
selection only. Measured occupancy, wave residency, cache behavior, and DRAM
transactions are unconfirmed because no profiler collection was run.
