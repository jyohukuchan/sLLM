# ullm-quant

`ullm-quant` is the prototype full-model aq converter.

The first skeleton only verifies the Rust CLI, C++20 kernel build, and C ABI boundary. Full safetensors input, calibration, and packed aq output will be added after the CPU kernel path is stable.

Initial target candidate:

```text
aq4_e4m3_g16_ts_flloyd16
```

Threading rules:

- `--threads` controls compute workers.
- `--io-threads` controls read/write helpers.
- Numeric kernels must not silently create another large thread pool.

