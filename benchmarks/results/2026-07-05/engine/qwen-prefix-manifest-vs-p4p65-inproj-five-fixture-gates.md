# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `5`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| p4p65-inproj | reject | 5 | -0.817754936 | -0.797094345 | 1.39915657 | fixture regression exceeds 0.001 |

## p4p65-inproj

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 1.78714752 | 1.14180946 | 10 | 0 | 3456 |
| tokens101 | 1.0805254 | 1.79110336 | 0.710577965 | 10 | 0 | 3456 |
| tokens201 | 1.140728 | 1.18086433 | 0.0401363373 | 7 | 0 | 3994 |
| tokens301 | 1.37130928 | 2.77046585 | 1.39915657 | 11 | 0 | 3456 |
| tokens401 | 0.959306717 | 1.75640106 | 0.797094345 | 10 | 0 | 3456 |
