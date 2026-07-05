# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `5`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| qwen35-hidden3994-policy | accept | 5 | 0.0479898453 | 0.022603035 | 0 | aggregate and median gates passed |

## qwen35-hidden3994-policy

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.629640579 | -0.0156974792 | 11 | 7 | 3994 |
| tokens101 | 1.0805254 | 1.0805254 | 0 | 7 | 12 | 3994 |
| tokens201 | 1.140728 | 1.00050735 | -0.140220642 | 11 | 13 | 3994 |
| tokens301 | 1.37130928 | 1.30988121 | -0.0614280701 | 10 | 12 | 3994 |
| tokens401 | 0.959306717 | 0.936703682 | -0.022603035 | 8 | 9 | 3994 |
