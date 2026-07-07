# Runtime cached prefix attention sweep

- schema: `runtime-cached-prefix-attn-sweep-v0.1`
- rows: 9

| status | executor | L | M | repeats | mean ms | new tok/s | pair/s | diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ok | cached_prefix_chunked | 4096 | 1 | 3 | 103.203890 | 9.689557 | 39698.115902 | 0.000000000 |
| ok | cached_prefix_chunked | 4096 | 16 | 3 | 124.240184 | 128.782810 | 528589.041882 | 0.000000000 |
| ok | cached_prefix_chunked | 4096 | 128 | 1 | 1019.441962 | 125.558889 | 522387.757078 | 0.000000000 |
| ok | cached_prefix_chunked | 16384 | 1 | 1 | 524.079953 | 1.908106 | 31264.313596 | 0.000000000 |
| ok | cached_prefix_chunked | 16384 | 16 | 1 | 578.890933 | 27.639058 | 453073.256202 | 0.000000000 |
| ok | cached_prefix_chunked | 16384 | 128 | 1 | 4616.811130 | 27.724764 | 456030.784175 | 0.000000000 |
| ok | cached_prefix_chunked | 65536 | 1 | 1 | 2055.547196 | 0.486488 | 31882.994527 | 0.000000000 |
| ok | cached_prefix_chunked | 65536 | 16 | 1 | 2356.189138 | 6.790626 | 445088.207516 | 0.000000000 |
| ok | cached_prefix_chunked | 65536 | 128 | 1 | 18234.605441 | 7.019620 | 460490.578048 | 0.000000000 |
