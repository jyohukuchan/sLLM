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
resident worker implementation is available, but the compatibility product
currently has no `promotion.json`, so AQ4 manifest generation remains
intentionally fail-closed until the resident path passes real-model validation
and a promotion receipt is published. Do not invent a receipt or placeholder
hash. Update `source_commit_from_receipt` only if the reviewed AQ4 receipt uses
a different field layout.

Generated manifests are ready for activation only after the corresponding
release worker, product, and promotion evidence pass the real-model release
gates. Generation proves file and contract identity; it does not prove runtime
correctness by itself.
