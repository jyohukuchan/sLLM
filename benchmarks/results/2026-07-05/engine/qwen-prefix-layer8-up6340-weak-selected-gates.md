# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `5`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| layer8-up6340-s1p004 | hold | 5 | -1.79290771e-05 | 0 | 0.000408172607 | median improvement below 0.005 or aggregate did not improve |
| layer8-up6340-s1p008 | hold | 5 | -3.83377075e-05 | 0 | 0.000799179077 | median improvement below 0.005 or aggregate did not improve |

## layer8-up6340-s1p004

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.645746231 | 0.000408172607 | 11 | 7 | 3994 |
| tokens101 | 1.0805254 | 1.0805254 | 0 | 7 | 12 | 3994 |
| tokens201 | 1.140728 | 1.1404171 | -0.000310897827 | 11 | 13 | 3994 |
| tokens301 | 1.37130928 | 1.37128448 | -2.47955322e-05 | 10 | 12 | 3994 |
| tokens401 | 0.959306717 | 0.959323883 | 1.71661377e-05 | 8 | 9 | 3994 |

## layer8-up6340-s1p008

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645338058 | 0.646137238 | 0.000799179077 | 11 | 7 | 3994 |
| tokens101 | 1.0805254 | 1.0805254 | 0 | 7 | 12 | 3994 |
| tokens201 | 1.140728 | 1.1401062 | -0.000621795654 | 11 | 13 | 3994 |
| tokens301 | 1.37130928 | 1.37129021 | -1.90734863e-05 | 10 | 12 | 3994 |
| tokens401 | 0.959306717 | 0.959340096 | 3.33786011e-05 | 8 | 9 | 3994 |
