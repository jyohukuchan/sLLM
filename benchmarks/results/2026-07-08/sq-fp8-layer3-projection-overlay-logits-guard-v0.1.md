# SQ FP8 Layer 3 Projection Overlay Logits Guard v0.1

## Summary

This is the first same-layer projection-set guard for the SQ FP8 overlay load path.

- Base package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d`
- SQ artifact: `/tmp/ullm-sq-fp8-layer3-projections-smoke`
- SQ candidate: `sq-fp8-w8a16-r9700-v0`
- Device: R9700/RDNA4, runtime device index `2`
- Layer: `3`
- Top-k: `8`
- LM head: AQ4 resident, group size `8`

The guard loads all selected self-attention and MLP projection tensors for layer 3 from SQ FP8,
while non-selected tensors fall back to the existing AQ4 package path.

## Overlay Tensors

- `model.language_model.layers.3.self_attn.q_proj.weight`
- `model.language_model.layers.3.self_attn.k_proj.weight`
- `model.language_model.layers.3.self_attn.v_proj.weight`
- `model.language_model.layers.3.self_attn.o_proj.weight`
- `model.language_model.layers.3.mlp.gate_proj.weight`
- `model.language_model.layers.3.mlp.up_proj.weight`
- `model.language_model.layers.3.mlp.down_proj.weight`

Artifact storage summary:

| field | value |
| --- | ---: |
| FP8 tensor count | `7` |
| passthrough tensor count | `768` |
| FP8 payload bytes | `209715200` |
| FP8 scale bytes | `172032` |
| compact resident bytes estimate | `19096673248` |
| materialized working-set bytes estimate | `201326592` |

## Commands

```bash
python3 tools/build-sq-fp8-w8a16-artifact.py \
  --source-model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B \
  --output-artifact /tmp/ullm-sq-fp8-layer3-projections-smoke \
  --base-package /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d \
  --include-regex '^model\.language_model\.layers\.3\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.weight$' \
  --overwrite
```

For each token sequence:

```bash
./target/debug/ullm-engine package-token-ids-logits-smoke \
  /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d \
  2 1048576 3 TOKEN_IDS_CSV 8 4096

./target/debug/ullm-engine sq-fp8-token-ids-logits-smoke \
  /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d \
  /tmp/ullm-sq-fp8-layer3-projections-smoke \
  2 1048576 3 TOKEN_IDS_CSV 8 4096
```

## Result

| case | token IDs | AQ4 top1 | SQ top1 | top1 match | top8 common | SQ layer load ms | SQ total ms |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| len4 | `1,2,3,4` | `55020` | `55020` | true | `7 / 8` | `18714.488662` | `19416.933020` |
| case_a | `100,200,300,400,500,600,700,800` | `15611` | `15611` | true | `5 / 8` | `18310.355575` | `19041.618644` |
| case_b | `42,314,2718,1618,12345,23456,34567,45678` | `54141` | `54141` | true | `4 / 8` | `18299.343782` | `19077.074536` |

- AQ4 baseline verified: true for all cases.
- SQ overlay verified: true for all cases.
- Top1 match: `3 / 3`.

## Interpretation

This moves T2 beyond a one-tensor overlay guard. The same-layer projection set can be loaded from
the SQ FP8 artifact without catastrophic short-logits quality collapse in this small guard bundle.

The timing is not an SQ performance result. The current overlay path decodes FP8 to host F32 and
copies a materialized F32 matrix into runtime buffers. The high SQ layer load time mainly measures
this temporary materialization path, not native FP8 execution.

## Status

This is still a partial T2 guard. It covers one full projection set for one self-attention layer.
It does not yet validate multiple layers, embedding/lm_head FP8 overlay, or full-target SQ candidate
quality.
