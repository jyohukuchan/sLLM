# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `3`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| layer6-mlp-selected | reject | 3 | 0.0120576223 | 0.00720596313 | 0.00403785706 | fixture regression exceeds 0.001 |

## layer6-mlp-selected

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.638132095 | -0.00720596313 | 11 | 7 | 3994 |
| tokens101 | 1.0805254 | 1.04752064 | -0.0330047607 | 7 | 12 | 3994 |
| tokens201 | 1.140728 | 1.14476585 | 0.00403785706 | 11 | 13 | 3994 |
