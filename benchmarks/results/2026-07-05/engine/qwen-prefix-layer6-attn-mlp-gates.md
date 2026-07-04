# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `3`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| layer6-attn-mlp | reject | 3 | 0.0115397771 | 0.00578689575 | 0.0117874146 | fixture regression exceeds 0.001 |

## layer6-attn-mlp

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.639551163 | -0.00578689575 | 11 | 7 | 3994 |
| tokens101 | 1.0805254 | 1.03990555 | -0.0406198502 | 7 | 12 | 3994 |
| tokens201 | 1.140728 | 1.15251541 | 0.0117874146 | 11 | 13 | 3994 |
