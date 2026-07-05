# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `2`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| targeted-high-layer8-qkv | reject | 2 | 0.0167789459 | 0.0167789459 | 0.00618362427 | fixture regression exceeds 0.001 |

## targeted-high-layer8-qkv

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.651521683 | 0.00618362427 | 11 | 7 | 3994 |
| tokens401 | 0.959306717 | 0.919565201 | -0.0397415161 | 8 | 9 | 3994 |
