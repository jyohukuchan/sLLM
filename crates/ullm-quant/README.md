# ullm-quant

`ullm-quant` is the prototype full-model aq converter.

The first skeleton verifies the Rust CLI, C++20 kernel build, and C ABI boundary.
It also reads safetensors metadata and writes a planning JSON without loading
tensor payloads.

Initial target candidate:

```text
aq4_e4m3_g16_ts_flloyd16
```

Threading rules:

- `--threads` controls compute workers.
- `--io-threads` controls read/write helpers.
- Numeric kernels must not silently create another large thread pool.

Planning example:

```text
cargo run -p ullm-quant -- \
  --model-dir /path/to/model \
  --plan-output /tmp/ullm-plan.json \
  --dry-run
```

The current default planner marks known text linear families as `quantize` and
leaves embeddings, lm head, vision tensors, convolution tensors, MTP tensors,
and unknown families as `passthrough`.
