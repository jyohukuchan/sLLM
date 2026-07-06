# Pre-SQ Runtime TPS Summary

| prompt | generated | device | decode_mode | prefill TPS | decode TPS | total wall seconds | KV cache bytes | verified |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | :---: |
| 1 | 1 | AMD Radeon Graphics (idx 2) | full_sequence_recompute_greedy | 0.075 | 0.074 | 13.515 | 0 | true |
| 1 | 2 | AMD Radeon Graphics (idx 2) | full_sequence_recompute_greedy | 0.074 | 0.068 | 28.272 | 0 | true |
| 1 | 2 | AMD Radeon Pro V620 (idx 1) | full_sequence_recompute_greedy | 0.071 | 0.068 | 29.054 | 0 | true |
| 16 | 4 | AMD Radeon Graphics (idx 2) | hybrid_incremental_greedy | 1.378 | 0.147 | 34.442 | 1310720 | true |
| 128 | 32 | AMD Radeon Graphics (idx 2) | hybrid_incremental_greedy | 2.900 | 0.141 | 265.797 | 10485760 | true |
| 128 | 32 | AMD Radeon Pro V620 (idx 1) | hybrid_incremental_greedy | 2.543 | 0.141 | 273.027 | 10485760 | true |
| 512 | 256 | AMD Radeon Graphics (idx 2) | hybrid_incremental_greedy | 2.909 | 0.139 | 2011.060 | 50331648 | true |
| 512 | 256 | AMD Radeon Pro V620 (idx 1) | hybrid_incremental_greedy | 2.507 | 0.140 | 2024.353 | 50331648 | true |

KV cache bytes are read from `memory.kv_cache_bytes` when present. Null values are estimated as f32 bytes with `cache_blocks * block_size * self_attention_layers * kv_heads * (head_dim + value_dim) * 4`.
