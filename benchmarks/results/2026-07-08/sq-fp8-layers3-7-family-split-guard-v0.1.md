# SQ FP8 Layers 3 and 7 Family Split Guard v0.1

## Summary

This guard splits the layers `3,7` SQ FP8 ranking drift by projection family.

- Base package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d`
- Source model: `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`
- Device: R9700/RDNA4, runtime device index `2`
- Layers: `3,7`
- Token IDs: `1,2,3,4`
- Top-k: `8`
- Raw JSON: `benchmarks/results/2026-07-08/sq-fp8-layers3-7-family-guard-v0.1.json`

The guard was run with `tools/run-sq-fp8-overlay-logits-guard.py`, which builds each SQ FP8
artifact, runs the AQ4 baseline once, runs each SQ overlay, and writes top-k comparison JSON.

## Result

AQ4 baseline top8:

`33604,239469,103346,49290,80054,35188,239762,148443`

| case | FP8 tensors | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit | layer load ms |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `q` | `2` | `239469` | false | `3` | `7 / 8` | `0.045671` | `6589.513054` |
| `k` | `2` | `33604` | true | `1` | `7 / 8` | `0.000000` | `1318.711837` |
| `v` | `2` | `239469` | false | `3` | `7 / 8` | `0.165494` | `1090.802837` |
| `o` | `2` | `33604` | true | `1` | `7 / 8` | `0.000000` | `3178.596847` |
| `gate` | `2` | `33604` | true | `1` | `8 / 8` | `0.000000` | `9308.762306` |
| `up` | `2` | `33604` | true | `1` | `8 / 8` | `0.000000` | `9311.913416` |
| `down` | `2` | `103346` | false | `2` | `8 / 8` | `0.039732` | `10067.597596` |

Top1 match count: `4 / 7`.

## Policy Subset Check

Raw JSON:

- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-policy-subset-guard-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-safe-subset-case-a-v0.1.json`
- `benchmarks/results/2026-07-08/sq-fp8-layers3-7-safe-subset-case-b-v0.1.json`

| case | coverage | FP8 tensors | SQ top1 | top1 match | AQ4 top1 rank in SQ top8 | top8 common | SQ top1 minus AQ4 top1 logit |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `safe_k_o_gate_up` | `k/o/gate/up` | `8` | `33604` | true | `1` | `7 / 8` | `0.000000` |
| `risk_q_v_down` | `q/v/down` | `6` | `239469` | false | `5` | `6 / 8` | `0.406537` |

Additional `safe_k_o_gate_up` prompt bundle:

| case | token IDs | AQ4 top1 | SQ top1 | top1 match | top8 common |
| --- | --- | ---: | ---: | --- | ---: |
| len4 | `1,2,3,4` | `33604` | `33604` | true | `7 / 8` |
| case_a | `100,200,300,400,500,600,700,800` | `15611` | `15611` | true | `5 / 8` |
| case_b | `42,314,2718,1618,12345,23456,34567,45678` | `227701` | `227701` | true | `7 / 8` |

Safe subset top1 match count across this short prompt bundle: `3 / 3`.

## Interpretation

Under the current row-scale FP8 E4M3 overlay:

- `k`, `o`, `gate`, and `up` are the safer families in this guard.
- `q`, `v`, and `down` are the first risky families for strict top1 preservation across layers
  `3,7`.
- Combining the safer families keeps AQ4 top1 unchanged in this short guard bundle.
- Combining the risky families moves AQ4 top1 to rank `5` and creates a much larger logit gap than
  any individual risky family.

The result does not prove that `k/o/gate/up` are safe for all layers or prompts. It does make them
the first strict-top1 subset worth expanding. The next T2 path is to keep strict-top1 qualification
for candidate expansion, and treat `q/v/down` as requiring a stronger scale/format policy or a
temporary non-FP8 fallback.

The timing still includes host-side FP8 to F32 materialization and should not be read as native FP8
throughput.

## Command

```bash
python3 tools/run-sq-fp8-overlay-logits-guard.py \
  --package /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d \
  --source-model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B \
  --output-root /tmp/ullm-sq-fp8-layers3-7-family-guard-v0.1 \
  --result-json benchmarks/results/2026-07-08/sq-fp8-layers3-7-family-guard-v0.1.json \
  --engine ./target/debug/ullm-engine \
  --device-index 2 \
  --chunk-bytes 1048576 \
  --layers 3,7 \
  --token-ids 1,2,3,4 \
  --top-k 8 \
  --lm-head-chunk-rows 4096 \
  --overwrite-artifacts \
  --case q='^model\.language_model\.layers\.(3|7)\.self_attn\.q_proj\.weight$' \
  --case k='^model\.language_model\.layers\.(3|7)\.self_attn\.k_proj\.weight$' \
  --case v='^model\.language_model\.layers\.(3|7)\.self_attn\.v_proj\.weight$' \
  --case o='^model\.language_model\.layers\.(3|7)\.self_attn\.o_proj\.weight$' \
  --case gate='^model\.language_model\.layers\.(3|7)\.mlp\.gate_proj\.weight$' \
  --case up='^model\.language_model\.layers\.(3|7)\.mlp\.up_proj\.weight$' \
  --case down='^model\.language_model\.layers\.(3|7)\.mlp\.down_proj\.weight$'
```
