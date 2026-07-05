# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `5`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| no-row-scale | reject | 5 | -0.630173302 | -0.494504929 | 1.13697243 | fixture regression exceeds 0.001 |

## no-row-scale

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 1.74426651 | 1.09892845 | 10 | 0 | 3456 |
| tokens101 | 1.0805254 | 1.50819016 | 0.427664757 | 10 | 0 | 3456 |
| tokens201 | 1.140728 | 1.13352394 | -0.00720405579 | 11 | 13 | 3994 |
| tokens301 | 1.37130928 | 2.50828171 | 1.13697243 | 11 | 0 | 3456 |
| tokens401 | 0.959306717 | 1.45381165 | 0.494504929 | 10 | 0 | 3456 |
