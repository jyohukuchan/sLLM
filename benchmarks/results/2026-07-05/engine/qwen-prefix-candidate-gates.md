# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `3`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| combined | needs_more_fixtures | 1 | 0.0343608856 | 0.0343608856 | 0 | only 1 paired fixture(s), need 3 |
| layer6 | reject | 3 | 0.013660113 | 0.0081653595 | 0.00455665588 | fixture regression exceeds 0.001 |

## combined

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.610977173 | -0.0343608856 | 11 | 7 | 3994 |

## layer6

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.637172699 | -0.0081653595 | 11 | 7 | 3994 |
| tokens101 | 1.0805254 | 1.04315376 | -0.0373716354 | 7 | 12 | 3994 |
| tokens201 | 1.140728 | 1.14528465 | 0.00455665588 | 11 | 13 | 3994 |
