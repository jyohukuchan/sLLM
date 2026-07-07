# Runtime cached prefix attention sweep

- schema: `runtime-cached-prefix-attn-sweep-v0.1`
- rows: 9

| status | executor | L | M | repeats | mean ms | new tok/s | pair/s | diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ok | cached_prefix_chunked | 4096 | 1 | 3 | 8.659609 | 115.478659 | 473116.067678 | 0.000000000 |
| ok | cached_prefix_chunked | 4096 | 16 | 3 | 8.994864 | 1778.792810 | 7301055.087249 | 0.000000000 |
| ok | cached_prefix_chunked | 4096 | 128 | 1 | 71.000950 | 1802.792780 | 7500519.359248 | 0.000000000 |
| ok | cached_prefix_chunked | 16384 | 1 | 1 | 40.262875 | 24.836776 | 406950.571712 | 0.000000000 |
| ok | cached_prefix_chunked | 16384 | 16 | 1 | 41.258902 | 387.795099 | 6356931.166031 | 0.000000000 |
| ok | cached_prefix_chunked | 16384 | 128 | 1 | 353.124919 | 362.477959 | 5962218.712750 | 0.000000000 |
| ok | cached_prefix_chunked | 65536 | 1 | 1 | 192.119321 | 5.205099 | 341126.543957 | 0.000000000 |
| ok | cached_prefix_chunked | 65536 | 16 | 1 | 196.368232 | 81.479575 | 5340537.974595 | 0.000000000 |
| ok | cached_prefix_chunked | 65536 | 128 | 1 | 1479.808982 | 86.497650 | 5674289.115783 | 0.000000000 |
