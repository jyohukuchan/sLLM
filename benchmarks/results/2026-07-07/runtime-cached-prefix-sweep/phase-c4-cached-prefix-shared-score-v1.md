# Runtime cached prefix attention sweep

- schema: `runtime-cached-prefix-attn-sweep-v0.1`
- rows: 9

| status | executor | L | M | repeats | mean ms | new tok/s | pair/s | diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ok | cached_prefix_chunked | 4096 | 1 | 3 | 47.865108 | 20.892045 | 85594.708602 | 0.000000000 |
| ok | cached_prefix_chunked | 4096 | 16 | 3 | 55.839281 | 286.536642 | 1176089.649006 | 0.000000000 |
| ok | cached_prefix_chunked | 4096 | 128 | 1 | 435.542576 | 293.886309 | 1222713.987897 | 0.000000000 |
| ok | cached_prefix_chunked | 16384 | 1 | 1 | 200.889494 | 4.977861 | 81562.254321 | 0.000000000 |
| ok | cached_prefix_chunked | 16384 | 16 | 1 | 303.245627 | 52.762509 | 864909.422090 | 0.000000000 |
| ok | cached_prefix_chunked | 16384 | 128 | 1 | 1748.356354 | 73.211619 | 1204221.322034 | 0.000000000 |
| ok | cached_prefix_chunked | 65536 | 1 | 1 | 911.030189 | 1.097658 | 71937.242905 | 0.000000000 |
| ok | cached_prefix_chunked | 65536 | 16 | 1 | 1738.452511 | 9.203588 | 603244.548450 | 0.000000000 |
| ok | cached_prefix_chunked | 65536 | 128 | 1 | 7959.297024 | 16.081822 | 1054975.580718 | 0.000000000 |
