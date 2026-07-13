# Served-model deployment profiles

The checked-in `*.profile.json` files contain model metadata and the WRX80
deployment paths, but deliberately contain no generated file hashes. Generate
an immutable `ullm.served_model.v1` document only after the release worker,
product, tokenizer, and promotion receipt are all in their final locations:

```bash
python3 tools/generate-served-model.py \
  --profile deploy/served-models/qwen3-14b-sq8.profile.json \
  --output /etc/ullm/served-models/qwen3-14b-sq8.json
python3 tools/validate-served-model.py \
  --manifest /etc/ullm/served-models/qwen3-14b-sq8.json
```

The generator streams SHA-256 calculation in 1 MiB chunks, extracts and hashes
the effective chat template from `tokenizer_config.json`, writes atomically,
and runs the gateway's strict validator before publishing the output. Re-run it
after every worker build; a checked-in hash would silently become stale.

The SQ8 profile can be materialized from the current promoted product. The AQ4
profile additionally requires an `ullm.aq4_resident_promotion.v1` receipt bound
to verified resident-vs-legacy evidence. Generate that evidence with
`tools/run-aq4-resident-promotion-evidence.py` and publish the receipt with
`tools/write-aq4-resident-promotion-receipt.py`. Do not invent a receipt or
placeholder hash.

`qwen35-9b-aq4-reasoning.profile.json` is an unactivated v2 candidate. It
declares the Qwen3.5 reasoning dialect and points to a separately built v2
worker under `target/reasoning-v2/`, but the current v1 promotion receipt is
still incompatible with it; generation must fail until matching v2 resident
evidence and a new receipt exist.

Generated manifests are ready for activation only after the corresponding
release worker, product, and promotion evidence pass the real-model release
gates. Generation proves file and contract identity; it does not prove runtime
correctness by itself.
