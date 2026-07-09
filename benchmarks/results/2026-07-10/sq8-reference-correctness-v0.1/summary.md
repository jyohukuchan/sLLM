# SQ8_0 Reference Correctness v0.1

Date: 2026-07-10

## Scope

This result validates one source-correct Qwen3-14B-FP8 projection through the P2
`reference_w8a16_block2d` path. It is a correctness result, not an optimized FP8 performance
claim.

- tensor: `model.layers.0.self_attn.q_proj.weight`
- shape: `5120 x 5120`
- scale grid: `40 x 40`
- block shape: `128 x 128`
- canonical content SHA-256: `29857be65d162ca1150f91e5c159186a0086d0f2eb463b2b113d92db4acd5c6a`
- fixed F32 activation SHA-256: `93f05449d07327c1237992938233030f1058dbe965504e343c8ae656dbe2e781`
- streaming oracle output SHA-256: `24661953d983d9532ef4c3413a420e73f139604762cc07e68eebd09da5b53469`

The oracle reads the FP8 weight with a fixed one-MiB buffer, keeps the compact BF16 block scale,
uses F64 accumulation, and emits F32 output. It does not materialize an F32 weight matrix.

## Frozen Gate

The gate was fixed before the formal runtime result:

- non-finite values: `0`
- max absolute error: `<= 2e-5`
- relative L2 error: `<= 1e-5`
- cosine similarity: `>= 0.999999`
- fallback: forbidden for the HIP result

## Results

| path | execution | fallback | max abs | relative L2 | cosine | result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| CPU runtime | `cpu_reference` | not applicable | `4.2915344e-6` | `7.8005394e-7` | `0.9999999999996875` | pass |
| R9700, enumeration index 2 / HIP device ID 1 | `hip_kernel` | not used | `2.3841858e-7` | `8.4994223e-8` | `0.9999999999999909` | pass |

The new block-2D HIP API returns an error when its native kernel is unavailable. It has no host
staging fallback branch. The R9700 row therefore proves execution of the scalar HIP reference
kernel, but does not claim matrix-core FP8 or optimized throughput.

Typed reports:

- [`cpu.json`](cpu.json)
- [`r9700.json`](r9700.json)

## Commands

```bash
cargo run --release --quiet -p ullm-engine --example sq8_reference_linear -- \
  --artifact /tmp/ullm-qwen3-14b-fp8-sq8-canonical-layer0-q-v0.2 \
  --tensor model.layers.0.self_attn.q_proj.weight \
  --device-index 0 \
  --report benchmarks/results/2026-07-10/sq8-reference-correctness-v0.1/cpu.json

cargo run --release --quiet -p ullm-engine --example sq8_reference_linear -- \
  --artifact /tmp/ullm-qwen3-14b-fp8-sq8-canonical-layer0-q-v0.2 \
  --tensor model.layers.0.self_attn.q_proj.weight \
  --device-index 2 \
  --report benchmarks/results/2026-07-10/sq8-reference-correctness-v0.1/r9700.json
```

P2 is green for one projection. P3 must now select and prove an optimized R9700 FP8 execution
route before P4 performance work is promoted.
