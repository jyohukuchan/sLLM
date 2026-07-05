# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `5`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| layer8-gateup | reject | 5 | -0.000522994995 | 0 | 0.0102748871 | fixture regression exceeds 0.001 |
| layer8-gateupfit | reject | 5 | -0.000232887268 | 0 | 0.00478172302 | fixture regression exceeds 0.001 |
| layer8-up | reject | 5 | -0.000444793701 | 0 | 0.00839996338 | fixture regression exceeds 0.001 |
| layer8-upfit | reject | 5 | -0.000185966492 | 0 | 0.00354194641 | fixture regression exceeds 0.001 |

## layer8-gateup

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.655612946 | 0.0102748871 | 11 | 7 | 3994 |
| tokens101 | 1.0805254 | 1.0805254 | 0 | 7 | 12 | 3994 |
| tokens201 | 1.140728 | 1.13283539 | -0.00789260864 | 11 | 13 | 3994 |
| tokens301 | 1.37130928 | 1.37111855 | -0.000190734863 | 10 | 12 | 3994 |
| tokens401 | 0.959306717 | 0.959730148 | 0.000423431396 | 8 | 9 | 3994 |

## layer8-gateupfit

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.650119781 | 0.00478172302 | 11 | 7 | 3994 |
| tokens101 | 1.0805254 | 1.0805254 | 0 | 7 | 12 | 3994 |
| tokens201 | 1.140728 | 1.13702393 | -0.00370407104 | 11 | 13 | 3994 |
| tokens301 | 1.37130928 | 1.37119675 | -0.000112533569 | 10 | 12 | 3994 |
| tokens401 | 0.959306717 | 0.959506035 | 0.000199317932 | 8 | 9 | 3994 |

## layer8-up

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.653738022 | 0.00839996338 | 11 | 7 | 3994 |
| tokens101 | 1.0805254 | 1.0805254 | 0 | 7 | 12 | 3994 |
| tokens201 | 1.140728 | 1.13436317 | -0.00636482239 | 11 | 13 | 3994 |
| tokens301 | 1.37130928 | 1.37115479 | -0.000154495239 | 10 | 12 | 3994 |
| tokens401 | 0.959306717 | 0.95965004 | 0.000343322754 | 8 | 9 | 3994 |

## layer8-upfit

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.648880005 | 0.00354194641 | 11 | 7 | 3994 |
| tokens101 | 1.0805254 | 1.0805254 | 0 | 7 | 12 | 3994 |
| tokens201 | 1.140728 | 1.13804817 | -0.00267982483 | 11 | 13 | 3994 |
| tokens301 | 1.37130928 | 1.37123108 | -7.82012939e-05 | 10 | 12 | 3994 |
| tokens401 | 0.959306717 | 0.959452629 | 0.00014591217 | 8 | 9 | 3994 |
