# Qwen Row-Scale Manifest Package Build

- schema: `qwen-row-scale-manifest-package-build-v0.1`
- source package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`
- output package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer8upfit-layer10-rebuilt.ullm.d`
- dry run: `False`
- row scale entries: `5`

| tensor | row | scale | source |
| --- | ---: | ---: | --- |
| model.language_model.layers.6.linear_attn.out_proj.weight | 3456 | 1.03227336478 | golden-prefix-row-dot-sensitivity-layer6 |
| model.language_model.layers.6.mlp.down_proj.weight | 3456 | 1.03658567925 | golden-prefix-row-dot-sensitivity-layer6 |
| model.language_model.layers.8.mlp.up_proj.weight | 6340 | 1.03510207319 | golden-prefix-row-dot-all-token-fit-layer8-token7-hidden3994-mlp-up |
| model.language_model.layers.10.linear_attn.out_proj.weight | 3456 | 1.0230717931 | golden-prefix-row-dot-sensitivity-layer10 |
| model.language_model.layers.10.mlp.down_proj.weight | 3456 | 1.04165701172 | golden-prefix-row-dot-sensitivity-layer10 |
