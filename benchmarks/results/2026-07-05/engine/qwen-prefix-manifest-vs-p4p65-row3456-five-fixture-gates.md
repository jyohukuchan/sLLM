# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `5`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| p4p65-row3456 | reject | 5 | -0.0887590408 | -0.0370130539 | 0.26203537 | fixture regression exceeds 0.001 |

## p4p65-row3456

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.790296555 | 0.144958496 | 7 | 0 | 3994 |
| tokens101 | 1.0805254 | 1.11753845 | 0.0370130539 | 6 | 0 | 3994 |
| tokens201 | 1.140728 | 1.13527489 | -0.00545310974 | 7 | 0 | 3994 |
| tokens301 | 1.37130928 | 1.37655067 | 0.00524139404 | 7 | 0 | 3994 |
| tokens401 | 0.959306717 | 1.22134209 | 0.26203537 | 9 | 5 | 3994 |
