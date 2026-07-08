# SQ FP8 Q Projection Overlay Logits Guard v0.1

## Summary

This is the first short logits guard using the SQ FP8 overlay load path.

- Base package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d`
- SQ artifact: `/tmp/ullm-sq-fp8-qproj-layer3-smoke`
- SQ candidate: `sq-fp8-w8a16-r9700-v0`
- Overlay tensor: `model.language_model.layers.3.self_attn.q_proj.weight`
- Device: R9700/RDNA4, runtime device index `2`
- Layers: `3`
- Input token IDs: `1,2,3,4`
- Top-k: `8`
- LM head: AQ4 resident, group size `8`

The guard proves the package logits path can consume selected FP8 SQ tensors while falling back to
the existing AQ4 package tensors for the rest of the layer.

## Commands

```bash
python3 tools/build-sq-fp8-w8a16-artifact.py \
  --source-model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B \
  --output-artifact /tmp/ullm-sq-fp8-qproj-layer3-smoke \
  --base-package /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d \
  --include-regex '^model\.language_model\.layers\.3\.self_attn\.q_proj\.weight$' \
  --overwrite

./target/debug/ullm-engine package-token-ids-logits-smoke \
  /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d \
  2 1048576 3 1,2,3,4 8 4096

./target/debug/ullm-engine sq-fp8-token-ids-logits-smoke \
  /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d \
  /tmp/ullm-sq-fp8-qproj-layer3-smoke \
  2 1048576 3 1,2,3,4 8 4096
```

## Result

| row | top1 | top8 token IDs |
| --- | ---: | --- |
| AQ4 baseline | `55020` | `55020,49290,25820,65471,226120,97947,103346,212237` |
| SQ FP8 q_proj overlay | `55020` | `55020,49290,25820,65471,212237,103346,226120,146557` |

- Top1 match: true
- Top8 common tokens: `7 / 8`
- Baseline verified: true
- SQ overlay verified: true

## Status

This is a partial T2 short guard. It validates one selected FP8 tensor overlay plus AQ4 fallback.
It does not yet validate a full FP8 SQ candidate covering all target projection tensors.
