# Qwen Cross-Fixture Tokens101-116 Summary

## Fixture

- Fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens101-116`
- Export command:
  - `tools/export-qwen-golden-tensors.py --model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B --token-ids 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116 --layer-range 0:12 --output benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens101-116 --device cpu --dtype bfloat16`
- Package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`
- Run mode: `actual_prefix`
- Layers: `0..12`
- Rotary: `rotary_dim=64`, `rope_base=10000000`, `position_offset=0`
- Backend: CPU

## Reports

- Baseline:
  - `package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens101-116-rotary64-manifest-row-scale-layer6-layer10-p4p46-inproj.jsonl`
- Layer6 row-scale:
  - `package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens101-116-rotary64-layer6h3994-row-scale-p4p46-inproj.jsonl`
- Layer8 QKV V845 cell:
  - `package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens101-116-rotary64-layer8qkv-v845-cell-p4p46-inproj.jsonl`
- Combined layer6 row-scale + layer8 QKV V845 cell:
  - `package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens101-116-rotary64-combined-layer6h3994-layer8qkv-p4p46-inproj.jsonl`

## Overall Results

| variant | overall max_abs | max layer | max token/hidden | layer11 max_abs |
| --- | ---: | ---: | --- | ---: |
| baseline | 1.080525398 | 7 | token 12 / hidden 3994 | 0.946708679 |
| layer6 row-scale | 1.043153763 | 7 | token 12 / hidden 3994 | 0.916080475 |
| layer8 QKV V845 cell | 1.080525398 | 7 | token 12 / hidden 3994 | 0.969970703 |
| combined | 1.043153763 | 7 | token 12 / hidden 3994 | 0.939382553 |

## Layer Detail

| variant | layer6 | layer7 | layer8 | layer9 | layer10 | layer11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.714679718 | 1.080525398 | 0.664204359 | 0.704177856 | 0.720614433 | 0.946708679 |
| layer6 row-scale | 0.652941704 | 1.043153763 | 0.664462686 | 0.730661392 | 0.720210314 | 0.916080475 |
| layer8 QKV V845 cell | 0.714679718 | 1.080525398 | 0.664207458 | 0.710563660 | 0.720012665 | 0.969970703 |
| combined | 0.652941704 | 1.043153763 | 0.664464235 | 0.737066269 | 0.719606876 | 0.939382553 |

## Layer6 Local Trace

Additional baseline input-dump and trace artifacts:

- input-dump smoke:
  - `package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens101-116-rotary64-baseline-input-dump-sample-t12-p4p46-inproj.jsonl`
- fullref trace:
  - `qwen-layer-module-trace-actual-input-rotary64-layer6-token12-hidden3994-tokens101-116-baseline-p4p46-inproj.jsonl`
- comparison:
  - `qwen-module-trace-comparison-actual-input-rotary64-layer6-token12-hidden3994-tokens101-116-baseline-p4p46-inproj.json`

Layer6 token12 hidden3994 local decomposition:

| component | value |
| --- | ---: |
| package output diff vs fixture | 0.403383 |
| package delta | -0.405800 |
| fullref delta on package input | -0.496683 |
| local delta error | 0.090883 |
| attention row-only / activation-path | 0.000485 / 0.012006 |
| MLP row-only / activation-path | -0.011936 / 0.106516 |

Layer6 `mlp.down_proj.weight[3994]` row-dot all-token fit:

| fixture | scale | row-dot RMSE before | row-dot RMSE after |
| --- | ---: | ---: | ---: |
| original token ids 1..16 | 1.026471714 | 0.117735388 | 0.063680278 |
| tokens101-116 | 1.023383096 | 0.131756300 | 0.061972585 |

Largest tokens101-116 package-source row-dot errors for the same row:

| token | package-source row-dot error |
| ---: | ---: |
| 0 | -0.510483047 |
| 13 | -0.062066241 |
| 6 | 0.056951314 |
| 15 | 0.051798975 |

## Interpretation

- The layer6 hidden3994 MLP down row-scale partially generalizes to this genuinely different token fixture:
  - overall max improves from `1.080525398` to `1.043153763`
  - layer6 max improves from `0.714679718` to `0.652941704`
  - layer7 max improves from `1.080525398` to `1.043153763`
- The layer6 local trace supports the same direction of row compensation:
  - tokens101-116 all-token LS scale is `1.023383096`, close to the original fixture scale `1.026471714`
  - the largest package-source row-dot error on this row is token `0`, `-0.510483047`
- The layer8 QKV V845 single-cell correction does not improve the overall max on this fixture:
  - overall max remains `1.080525398`
  - layer11 worsens from `0.946708679` to `0.969970703`
- Combined equals layer6 row-scale on the overall max, but the QKV cell shifts later-layer errors:
  - layer11 is `0.939382553`, better than baseline but worse than layer6 row-scale alone.
- Current durable-fix priority:
  - Promote or replace the layer6 row-scale as a quantizer/package policy candidate.
  - Treat the layer8 QKV V845 cell as fixture-specific until more fixtures support it.
