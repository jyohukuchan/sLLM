# Qwen Row-Scale Candidates

- schema: `qwen-row-scale-candidates-v0.1`
- trace count: `1`
- observation count: `2`
- candidate count: `2`

| layer | tensor_suffix | row | scale | selected | obs | rmse | scaled_rmse | improvement |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 8 | linear_attn.out_proj.weight | 3994 | 0.972727612 | tokens401-layer8 | 1 | 0.0576938288 | 0.041785554 | 0.275736161 |
| 8 | mlp.down_proj.weight | 3994 | 1.01213864 | tokens401-layer8 | 1 | 0.0240479274 | 0.0191838603 | 0.202265545 |
