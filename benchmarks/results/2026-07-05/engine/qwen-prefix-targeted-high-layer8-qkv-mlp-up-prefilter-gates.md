# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `2`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| targeted-high-layer8-qkv-mlp-up | accept | 2 | 0.0191502571 | 0.0191502571 | 0 | aggregate and median gates passed |

## targeted-high-layer8-qkv-mlp-up

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.629640579 | -0.0156974792 | 11 | 7 | 3994 |
| tokens401 | 0.959306717 | 0.936703682 | -0.022603035 | 8 | 9 | 3994 |
