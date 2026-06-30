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

Policy presets:

- `all-g16`: all quantizable tensors use the low-budget format.
- `all-g8`: all quantizable tensors use the high-budget format.
- `p4p6`: `attn_k`, `attn_o`, `attn_v`, and `linear_attn_out` use the
  high-budget format; other quantizable families use the low-budget format.
- `p4p9`: `attn_k`, `attn_o`, `attn_q`, `attn_v`, `linear_attn_out`,
  `mlp_gate`, and `mlp_up` use the high-budget format.
- `custom`: use repeated `--aq-high-family <FAMILY>`.
