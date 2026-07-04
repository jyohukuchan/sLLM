# Qwen Row-Scale Candidates

- schema: `qwen-row-scale-candidates-v0.1`
- trace count: `3`
- observation count: `4`
- candidate count: `3`

| layer | tensor_suffix | row | scale | selected | obs | rmse | scaled_rmse | improvement |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 6 | linear_attn.out_proj.weight | 3994 | 0.992316548 | tokens101-layer6 | 1 | 0.0385251718 | 0.0267212642 | 0.306394678 |
| 6 | mlp.down_proj.weight | 3994 | 1.0233831 | tokens101-layer6 | 2 | 0.1317563 | 0.0619725846 | 0.529642343 |
| 11 | self_attn.o_proj.weight | 3994 | 0.984954853 | tokens201-layer11 | 1 | 0.0598555195 | 0.0241662081 | 0.59625765 |
