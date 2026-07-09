# SQ8 CK optimized component benchmark

This standalone P4 benchmark measures dynamic row-by-K128 OCP E4M3 activation quantization plus Composable Kernel ABScale GEMM on an isolated gfx1201 device. It does not modify the production runtime dispatch.

Build once:

```bash
tools/run-sq8-ck-component.sh --build-only
```

Run one fixture through the wrapper:

```bash
ULLM_R9700_HIP_VISIBLE_DEVICE=1 tools/run-sq8-ck-component.sh \
  --m 8 --n 5120 --k 5120 \
  --weight /path/to/weight.f8_e4m3 \
  --weight-scale /path/to/weight_scale.bf16 \
  --activation /path/to/activation.f32le \
  --oracle /path/to/oracle_output.f32le \
  --expected-activation-fp8 /path/to/activation.f8 \
  --expected-activation-scales /path/to/activation_scales.f32le \
  --warmups 10 --repeats 50
```

To rule out a result that depends on the 26 MiB weight remaining in cache, add
`--cache-mode evicted`. Before every GEMM sample, the benchmark reads a separate
256 MiB hash-initialized GPU buffer on the same stream. The eviction dispatch is
ordered before the start event and excluded from the reported duration. Its
checksum, validation duration, byte count, and device L2 size are recorded in JSON.

The wrapper requires exactly one HIP visibility token and maps it to internal device zero. The result is one JSON document. A passing result requires exact activation FP8 bytes and scale bits when expected files are supplied, finite BF16 output, relative L2 at most `5e-3`, cosine similarity at least `0.9999`, a numerically valid CK candidate, and no fallback.

`quant_only`, `gemm_only`, and `quant_plus_gemm` are HIP-event timings. Warm mode
repeats the same resident buffers; evicted mode replaces target data from cache
before each GEMM measurement while leaving `quant_only` warm. The promotion number
is `quant_plus_gemm`; model load, file I/O, host-to-device weight upload, and cache
eviction are outside the timed region. A 2 GiB estimated host/device working-set
budget and an available-device-memory check reject accidental oversized dimensions
before input files are allocated.
