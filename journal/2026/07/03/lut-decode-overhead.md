# LUT decode overhead benchmark

## Context

`k <= 7` の kbit codebook-index を FP8 payload へ、`n <= 12` の nbit codebook-index を FP16 payload へ LUT で置き換える場合の変換 overhead を測った。君の補足に合わせて、LUT を register や shared memory へ明示配置することは考えず、通常の配列参照を hardware cache に任せる前提にした。

測定対象は payload 変換であり、FP8/FP16 の算術変換ではない。FP8 は `uint8_t`、FP16 は `uint16_t` の payload として扱う。

## Tools

- CPU: `tools/bench-lut-decode.cpp`
- HIP: `tools/bench-lut-decode-hip.cpp`

mode:

- `store_only`: output store の baseline
- `aligned_index_lut`: `uint16_t` index から `lut[index]` を出力
- `packed_index_lut`: packed bitstream から index を取り出して `lut[index]` を出力

## R9700/gfx1201 Results

Command:

```bash
build/tools/bench-lut-decode-hip --device 1 --values 67108864 --repeats 20 --warmups 5
```

### FP8 payload, packed_index_lut

store-only baseline: `0.003101230 ns/value`

| k | LUT bytes | ns/value | overhead vs store | Gvalues/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 0.007502 | 0.004401 | 133.300 |
| 2 | 4 | 0.007947 | 0.004845 | 125.841 |
| 3 | 8 | 0.008062 | 0.004961 | 124.036 |
| 4 | 16 | 0.008186 | 0.005084 | 122.167 |
| 5 | 32 | 0.008375 | 0.005274 | 119.402 |
| 6 | 64 | 0.007874 | 0.004773 | 127.003 |
| 7 | 128 | 0.007909 | 0.004808 | 126.439 |

### FP16 payload, packed_index_lut

store-only baseline: `0.003395081 ns/value`

| n | LUT bytes | ns/value | overhead vs store | Gvalues/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 0.007795 | 0.004400 | 128.285 |
| 2 | 8 | 0.007839 | 0.004444 | 127.573 |
| 3 | 16 | 0.007865 | 0.004470 | 127.138 |
| 4 | 32 | 0.008382 | 0.004987 | 119.308 |
| 5 | 64 | 0.008056 | 0.004661 | 124.128 |
| 6 | 128 | 0.008355 | 0.004960 | 119.691 |
| 7 | 256 | 0.008494 | 0.005099 | 117.726 |
| 8 | 512 | 0.008438 | 0.005043 | 118.516 |
| 9 | 1024 | 0.008764 | 0.005369 | 114.107 |
| 10 | 2048 | 0.008829 | 0.005434 | 113.260 |
| 11 | 4096 | 0.008755 | 0.005360 | 114.216 |
| 12 | 8192 | 0.009068 | 0.005673 | 110.282 |

## V620/gfx1030 Summary

V620 も同じ HIP binary で測った。

- FP8 packed: `0.007696-0.008143 ns/value`
- FP16 packed: `0.009850-0.011277 ns/value`

FP8 は R9700 とほぼ同じ範囲、FP16 は V620 のほうが少し遅い。

## CPU Summary

TR PRO 3995WX の single-thread C++ scalar path も測った。

- FP8 packed: `0.904781-1.149093 ns/value`
- FP16 packed: `0.589038-1.947172 ns/value`

CPU 側は scalar bit unpack の性質が強く出るので、GPU hot path の判断材料としては HIP 結果を優先する。

## Interpretation

R9700 では FP8 の k=1..7 はほぼ横ばいだった。LUT size は最大 128 bytes なので、cache 前提なら k を増やしても LUT lookup 自体の penalty は見えにくい。packed input は aligned `uint16_t` index より読み込み量が小さいため、`packed_index_lut` が `aligned_index_lut` より速いケースもある。

FP16 の n=1..12 は少しずつ遅くなるが、n=12 でも LUT size は 8 KiB で、R9700 の packed path は `0.009068 ns/value` だった。n=1 から n=12 への増加は約 16% で、急激な cache cliff は出ていない。

この測定は単体 kernel の aggregate ns/value であり、GEMM 内に fused したときの命令スケジューリングやメモリアクセス競合は含まない。次に必要なのは、実際の dequant + dot/GEMM kernel 内で `packed_index_lut` を使った場合の token/s または effective bandwidth 測定。

## Artifacts

- summary: `benchmarks/results/2026-07-03/aq/2026-07-03-lut-decode-summary.json`
- CPU raw: `benchmarks/results/2026-07-03/aq/2026-07-03-lut-decode-cpu-values16m.json`
- R9700 raw: `benchmarks/results/2026-07-03/aq/2026-07-03-lut-decode-hip-r9700-values64m.json`
- V620 raw: `benchmarks/results/2026-07-03/aq/2026-07-03-lut-decode-hip-v620-values64m.json`
