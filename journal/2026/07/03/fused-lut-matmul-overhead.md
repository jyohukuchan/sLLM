# Fused LUT matmul overhead benchmark

## Context

前回の standalone LUT decode benchmark は、packed codebook-index を FP8/FP16 payload へ materialize する処理を測っていた。これは実推論の hot path とは違うので、今回は fused dequant に近い形で測った。

比較:

- baseline: expanded FP8/FP16 payload weight を読み、同じ scalar dot kernel で使う
- fused: packed codebook-index を読み、global-memory LUT から FP8/FP16 payload を取り、同じ scalar dot kernel で即使う

君の補足に合わせて、LUT は普通の global-memory 配列として置き、hardware cache に任せた。LUT を register/shared memory へ明示配置する測定はしていない。

注意:

- この benchmark は MFMA/Tensor-core GEMM ではない。
- FP8 は E4M3 payload を software decode して dot に使うため、最終的な FP8 hardware GEMM の速度ではない。
- ただし standalone materialize よりは、実際の fused dequant overhead の判断に近い。

## Tool

- `tools/bench-fused-lut-matmul-hip.cpp`

## R9700/gfx1201 Results

### M=1, N=4096, K=4096

| target | bits | baseline ms | fused ms | ratio |
| --- | ---: | ---: | ---: | ---: |
| FP8 payload | 4 | 0.089121 | 0.111601 | 1.252 |
| FP8 payload | 7 | 0.089041 | 0.117162 | 1.316 |
| FP16 payload | 4 | 0.072361 | 0.097601 | 1.349 |
| FP16 payload | 8 | 0.072561 | 0.101441 | 1.398 |
| FP16 payload | 12 | 0.072561 | 0.107361 | 1.480 |

### M=1, N=12288, K=4096

| target | bits | baseline ms | fused ms | ratio |
| --- | ---: | ---: | ---: | ---: |
| FP8 payload | 4 | 0.250363 | 0.311444 | 1.244 |
| FP8 payload | 7 | 0.251004 | 0.326324 | 1.300 |
| FP16 payload | 4 | 0.280924 | 0.272003 | 0.968 |
| FP16 payload | 8 | 0.282564 | 0.294204 | 1.041 |
| FP16 payload | 12 | 0.283124 | 0.403085 | 1.424 |

FP16 payload では n=1..10 の範囲で fused が baseline と同等または速い行が出た。expanded FP16 weight は 2 bytes/value だが、packed nbit は n/8 bytes/value なので、bandwidth 減少が LUT/bit unpack cost を相殺していると考える。

### M=16, N=4096, K=4096

| target | bits | baseline ms | fused ms | ratio |
| --- | ---: | ---: | ---: | ---: |
| FP8 payload | 4 | 1.277935 | 1.429378 | 1.119 |
| FP8 payload | 7 | 1.278696 | 1.651260 | 1.291 |
| FP16 payload | 4 | 1.054854 | 1.227815 | 1.164 |
| FP16 payload | 8 | 1.055093 | 1.526099 | 1.446 |
| FP16 payload | 12 | 1.044453 | 1.605661 | 1.537 |

## Summary

R9700 の fused overhead は、shape により次の範囲だった。

| run | target | fused/baseline ratio range | overhead ns/weight-use range |
| --- | --- | ---: | ---: |
| M1 4096x4096 | FP8 | 1.160-1.317 | 0.000854-0.001678 |
| M1 4096x4096 | FP16 | 1.208-1.480 | 0.000899-0.002074 |
| M1 12288x4096 | FP8 | 1.144-1.300 | 0.000714-0.001496 |
| M1 12288x4096 | FP16 | 0.855-1.424 | -0.000811-0.002383 |
| M16 4096x4096 | FP8 | 1.100-1.338 | 0.000473-0.001615 |
| M16 4096x4096 | FP16 | 1.123-1.537 | 0.000481-0.002091 |

V620/gfx1030 の representative M=1,N=4096,K=4096 も測ったが、bits により fused が大きく速くなる行と遅くなる行が混在した。scalar kernel と memory pattern の影響が大きいので、V620 は参考値として扱う。

## Interpretation

10B parameter を materialize すると 80ms 級という前回試算は、実推論 overhead としては不適切だった。今回の fused dot では、R9700 で `0.0005-0.0024 ns/weight-use` 程度の差分になっている。

ただし、この benchmark もまだ production GEMM ではない。最終判断には、次の段階として MFMA/WMMA 相当の tile kernel または実 engine 内の matmul path で、expanded FP8/FP16 と packed LUT fused を比較する必要がある。

## Artifacts

- summary: `benchmarks/results/2026-07-03/aq/2026-07-03-fused-lut-matmul-summary.json`
- R9700 raw:
  - `benchmarks/results/2026-07-03/aq/2026-07-03-fused-lut-matmul-hip-r9700-m1-n4096-k4096.json`
  - `benchmarks/results/2026-07-03/aq/2026-07-03-fused-lut-matmul-hip-r9700-m1-n12288-k4096.json`
  - `benchmarks/results/2026-07-03/aq/2026-07-03-fused-lut-matmul-hip-r9700-m16-n4096-k4096.json`
- V620 raw:
  - `benchmarks/results/2026-07-03/aq/2026-07-03-fused-lut-matmul-hip-v620-m1-n4096-k4096.json`
