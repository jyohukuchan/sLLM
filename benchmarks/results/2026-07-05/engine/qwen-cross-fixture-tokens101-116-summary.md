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

## Interpretation

- The layer6 hidden3994 MLP down row-scale partially generalizes to this genuinely different token fixture:
  - overall max improves from `1.080525398` to `1.043153763`
  - layer6 max improves from `0.714679718` to `0.652941704`
  - layer7 max improves from `1.080525398` to `1.043153763`
- The layer8 QKV V845 single-cell correction does not improve the overall max on this fixture:
  - overall max remains `1.080525398`
  - layer11 worsens from `0.946708679` to `0.969970703`
- Combined equals layer6 row-scale on the overall max, but the QKV cell shifts later-layer errors:
  - layer11 is `0.939382553`, better than baseline but worse than layer6 row-scale alone.
- Current durable-fix priority:
  - Promote or replace the layer6 row-scale as a quantizer/package policy candidate.
  - Treat the layer8 QKV V845 cell as fixture-specific until more fixtures support it.
