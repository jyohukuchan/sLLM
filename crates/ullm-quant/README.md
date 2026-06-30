# ullm-quant

`ullm-quant` is the prototype full-model aq converter.

The first skeleton verifies the Rust CLI, C++20 kernel build, and C ABI boundary.
It also reads safetensors metadata and writes a planning JSON without loading
tensor payloads.

Initial low-budget target candidate:

```text
aq4_e4m3_g16_ts_flloyd16
```

Initial high-budget target candidate:

```text
aq4_e4m3_g8_ts_flloyd16
```

Threading rules:

- `--threads` controls compute workers.
- `--io-threads` controls read/write helpers.
- Numeric kernels must not silently create another large thread pool.

Planning example:

```text
cargo run -p ullm-quant -- \
  --model-dir /path/to/model \
  --aq-policy p4p6 \
  --plan-output /tmp/ullm-plan.json \
  --dry-run
```

The current default planner marks known text linear families as `quantize` and
leaves embeddings, lm head, vision tensors, convolution tensors, MTP tensors,
and unknown families as `passthrough`.
It also records an index+scale payload-size estimate for quantized tensors.
The next conversion step should use the existing safetensors chunk reader so
payload processing stays bounded by the working-memory budget.

Chunk inspection example:

```text
cargo run -p ullm-quant -- \
  --model-dir /path/to/model \
  --inspect-tensor model.language_model.layers.0.mlp.up_proj.weight \
  --chunk-bytes 1048576 \
  --dry-run
```

For BF16 and F32 tensors, inspection also reports chunked numeric stats such as
min, max, mean absolute value, and max absolute value.
With `--inspect-aq-format`, it also reports group-count and group-absmax stats
using the aq candidate's group size.
For scale-format dry runs, it reports direct group-absmax scale index range,
clamp counts, and mean relative scale error. This is a range check, not the
final quantizer scale search.

Tensor-scale estimation defaults to `--tensor-scale-estimator exact`, which
keeps all positive group target scales so it can use the exact lower median.
For lower-memory prototype runs, use `--tensor-scale-estimator reservoir` and
optionally `--tensor-scale-reservoir-size <N>`. Reservoir mode is deterministic
and bounds stored target scales to `N`, but it is an approximate estimator and
should be compared against exact mode before changing production defaults.

Codebook inspection example:

```text
cargo run -p ullm-quant -- \
  --codebook-json /path/to/codebooks.json \
  --inspect-codebook-family mlp_up \
  --inspect-codebook-candidate aq4_e4m3_g16_ts_flloyd16 \
  --dry-run
```

Multi-tensor prototype conversion example:

```text
cargo run -p ullm-quant --release -- \
  --convert-plan-json /path/to/ullm-plan.json \
  --codebook-json /path/to/codebooks.json \
  --convert-output-root /tmp/model.ullm.parts \
  --convert-summary-output /tmp/model-convert-summary.json \
  --convert-family mlp_up \
  --convert-family attn_k \
  --convert-max-tensors 4 \
  --convert-per-family 2 \
  --convert-jobs 2 \
  --scale-window 4 \
  --tensor-scale-estimator reservoir \
  --tensor-scale-reservoir-size 65536 \
  --convert-verify \
  --convert-overwrite \
  --dry-run
```

This writes one prototype `.ullm.d` directory per selected quantized tensor.
Selection is driven by the plan's `action`, `family`, and `quant_format`; tensors
without matching exported codebooks are skipped. Use `--convert-family`,
`--convert-max-tensors`, and `--convert-per-family` for bounded smokes before
full conversion. `--convert-jobs` enables tensor-level parallelism, but the
default is `1` so conversion memory stays predictable unless a run explicitly
opts in.

Policy presets:

- `all-g16`: all quantizable tensors use the low-budget format.
- `all-g8`: all quantizable tensors use the high-budget format.
- `p4p6`: `attn_k`, `attn_o`, `attn_v`, and `linear_attn_out` use the
  high-budget format; other quantizable families use the low-budget format.
- `p4p9`: `attn_k`, `attn_o`, `attn_q`, `attn_v`, `linear_attn_out`,
  `mlp_gate`, and `mlp_up` use the high-budget format.
- `p4p46_inproj`: `attn_o`, `attn_v`, `linear_attn_a`, `linear_attn_b`,
  `linear_attn_out`, and `linear_attn_z` use the high-budget format.
- `p4p65_inproj`: `attn_k`, `attn_o`, `attn_v`, `linear_attn_a`,
  `linear_attn_b`, `linear_attn_out`, and `linear_attn_qkv` use the
  high-budget format.
- `custom`: use repeated `--aq-high-family <FAMILY>`.
