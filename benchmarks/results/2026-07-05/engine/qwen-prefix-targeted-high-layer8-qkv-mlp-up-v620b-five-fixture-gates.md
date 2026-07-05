# Qwen Prefix Candidate Gates

- schema: `qwen-prefix-candidate-gates-v0.1`
- baseline condition: `baseline`
- max fixture worsen: `0.001`
- min median improvement: `0.005`
- min fixture count: `5`

| condition | decision | fixtures | mean improvement | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| targeted-high-layer8-qkv-mlp-up | accept | 5 | 0.0479856491 | 0.0226106644 | 0 | aggregate and median gates passed |

## targeted-high-layer8-qkv-mlp-up

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | 0.645345688 | 0.629638672 | -0.015707016 | 11 | 7 | 3994 |
| tokens101 | 1.08053589 | 1.08053589 | 0 | 7 | 12 | 3994 |
| tokens201 | 1.140728 | 1.00051689 | -0.140211105 | 11 | 13 | 3994 |
| tokens301 | 1.37132263 | 1.30992317 | -0.0613994598 | 10 | 12 | 3994 |
| tokens401 | 0.959306717 | 0.936696053 | -0.0226106644 | 8 | 9 | 3994 |
