# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `3`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| extracted | reject | 3 | -0.0211502711 | -0.0312900543 | 0.0727806091 | fixture regression exceeds 0.001 |

## extracted

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.676628113 | 0.0312900543 | 11 | 7 | 3994 |
| tokens101 | 1.0805254 | 1.03990555 | -0.0406198502 | 7 | 12 | 3994 |
| tokens201 | 1.140728 | 1.21350861 | 0.0727806091 | 11 | 13 | 3994 |
