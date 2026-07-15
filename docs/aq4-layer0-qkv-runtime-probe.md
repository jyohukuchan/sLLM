# AQ4 layer-0 QKV runtime probe

`ullm-aq4-layer0-qkv-runtime-probe` is a diagnostic binary for one narrowly
defined operation: a direct `PackageAq4ResidentMatvec::matvec` call for
`model.language_model.layers.0.linear_attn.in_proj_qkv.weight`. The probe does
not call the fused QKV/Z/Gate/Beta wrapper. The report is therefore emitted
with `classification: "unclassified"` and `promotion_eligible: false`.

## Input sidecar

The input is UTF-8 JSON Lines. The first line is a strict header:

```json
{"kind":"header","schema_version":"ullm.aq4_layer0_input_normed_jsonl.v1","tensor_name":"model.language_model.layers.0.linear_attn.in_proj_qkv.weight","dtype":"f32","shape":[4096]}
```

Each following line is one already-materialized `input_normed` vector. Values
are serialized as JSON numbers, but the hash is over their exact little-endian
IEEE-754 f32 bytes (4096 values):

```json
{"kind":"case","case_id":"fixture-prompt-0","step":0,"context_token_ids_sha256":"<64 lowercase hex chars>","context_length":3,"input_sha256":"<sha256 of f32le values>","values":[...]}
```

The header shape, tensor name, dtype, case identity, finite values, input hash,
and context hash are fail-closed. Case IDs plus steps must be unique. Input
lines are capped and consumed sequentially to keep memory bounded.

## Running the CPU probe

Run only device 0 for the CPU oracle. The command below writes two new files
and refuses to overwrite either one:

```text
cargo run -p ullm-engine --bin ullm-aq4-layer0-qkv-runtime-probe -- \
  --package /path/to/package \
  --input /path/to/input.jsonl \
  --output-dir /path/to/new-output-dir \
  --device-index 0
```

`output.f32le` contains concatenated little-endian f32 rows in input order.
`report.json` binds the package manifest SHA, all three QKV payload SHAs,
tensor geometry, per-case context/input hashes, runtime device information,
operation identity, and relevant guard environment variables. The input
sidecar report records its canonical regular-file path, `nlink=1`, device,
inode, byte size, nanosecond mtime before and after consumption, and the SHA
of the bytes consumed by the probe. Any metadata or digest change fails
closed. Every output value must be finite. Each sidecar is written to a
temporary file, synced, and published with a no-overwrite hard link.

The same binary can target a diagnostic HIP device only through the separately
gated standalone probe gate, with `--device-index 1`,
`HIP_VISIBLE_DEVICES=1`, and `ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1`; fallback is
never allowed. This threshold-free path is not a numeric Go/No-Go, does not
observe holdout data, and always remains `unclassified` with
`promotion_eligible: false`. It must not be treated as fused-wrapper or serving
evidence.
