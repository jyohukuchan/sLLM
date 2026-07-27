# RDNA4 (R9700) / CDNA3 (MI300X) hardware microbenchmark

Status: R9700 measurement is pending exclusive GPU availability; MI300X is not
yet rented.  Empty fields are intentionally not zero.

| Item | R9700 (gfx1201) | MI300X (gfx942) |
|---|---|---|
| STREAM read | pending | pending |
| STREAM copy | pending | pending |
| STREAM triad | pending | pending |
| BF16 GEMM 256³ / 1024³ / 4096³ | pending | pending |
| FP8 GEMM 256³ / 1024³ / 4096³ | pending | pending |
| BF16 Qwen3-14B `256x5120x5120` | pending | pending |
| FP8 Qwen3-14B `256x5120x5120` | pending | pending |
| CPU numeric oracle | pending | pending (run on rental) |
| ISA | pending | offline pass: MFMA evidence in `isa/` |
| telemetry / clocks / power | pending | pending |

Metric definition: GEMM is dense `2*M*N*K` FLOPs and STREAM byte counts are
read=4N, copy=8N, triad=12N. Each reported throughput is the median of 11 HIP
event samples after 5 warmups; each sample contains 10 kernel launches.
