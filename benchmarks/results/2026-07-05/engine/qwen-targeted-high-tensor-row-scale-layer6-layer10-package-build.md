# Qwen Row-Scale Manifest Package Build

- schema: `qwen-row-scale-manifest-package-build-v0.1`
- source package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-mlp-up-high-reservoir65536-jobs64.ullm.d`
- output package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-mlp-up-high-row-scale-layer6-layer10.ullm.d`
- dry run: `False`
- row scale entries: `4`

| tensor | row | scale | source |
| --- | ---: | ---: | --- |
| model.language_model.layers.6.linear_attn.out_proj.weight | 3456 | 1.03227336478 | golden-prefix-row-dot-sensitivity-layer6 |
| model.language_model.layers.6.mlp.down_proj.weight | 3456 | 1.03658567925 | golden-prefix-row-dot-sensitivity-layer6 |
| model.language_model.layers.10.linear_attn.out_proj.weight | 3456 | 1.0230717931 | golden-prefix-row-dot-sensitivity-layer10 |
| model.language_model.layers.10.mlp.down_proj.weight | 3456 | 1.04165701172 | golden-prefix-row-dot-sensitivity-layer10 |
