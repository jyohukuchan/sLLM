# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `2`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| targeted-high-layer8-mlp-up | reject | 2 | 0.00118732452 | 0.00118732452 | 0.0153160095 | fixture regression exceeds 0.001 |

## targeted-high-layer8-mlp-up

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.6276474 | -0.0176906586 | 7 | 0 | 3994 |
| tokens401 | 0.959306717 | 0.974622726 | 0.0153160095 | 8 | 9 | 3994 |
