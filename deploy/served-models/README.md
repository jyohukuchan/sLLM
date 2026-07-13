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
not reused and the v2 receipt path is not yet published; generation must fail
until matching v2 resident evidence and a new receipt exist.

Rebuild that isolated candidate worker from the current checkout with:

```bash
cargo build --release --bin ullm-aq4-worker --target-dir target/reasoning-v2
```

After an exclusive R9700 window is available, keep the evidence and receipt in
the product directory and run the following sequence. The promotion runner
must finish before the receipt is published:

```bash
PRODUCT=/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1
EVIDENCE="$PRODUCT/resident-promotion-reasoning-v2-v0.1.json"
EVIDENCE_INCOMPLETE="$EVIDENCE.incomplete"
RECEIPT="$PRODUCT/promotion-reasoning-v2-v0.1.json"

uv run --project services/openai-gateway python \
  tools/run-aq4-resident-promotion-evidence.py \
  --profile deploy/served-models/qwen35-9b-aq4-reasoning.profile.json \
  --worker-binary target/reasoning-v2/release/ullm-aq4-worker \
  --legacy-engine target/release/ullm-engine \
  --output "$EVIDENCE_INCOMPLETE" && \
  mv -- "$EVIDENCE_INCOMPLETE" "$EVIDENCE"

uv run --project services/openai-gateway python \
  tools/write-aq4-resident-promotion-receipt.py \
  --profile deploy/served-models/qwen35-9b-aq4-reasoning.profile.json \
  --evidence "$EVIDENCE" \
  --output "$RECEIPT"
```

Only after those commands succeed may `generate-served-model.py` materialize
the v2 manifest. The existing active v1 receipt must not be overwritten.

Generated manifests are ready for activation only after the corresponding
release worker, product, and promotion evidence pass the real-model release
gates. Generation proves file and contract identity; it does not prove runtime
correctness by itself.

For the final hash-only handoff, keep the generic release evidence, its
validator report, the OpenWebUI browser evidence and report, and the promotion
evidence and receipt beside a `ullm.generic_reasoning_release_bundle.v1`
document. Validate that document with:

```bash
uv run --project services/openai-gateway python \
  tools/validate-generic-reasoning-release-bundle.py \
  /path/to/generic-reasoning-release-bundle.json \
  --require-complete
```

The bundle validator re-hashes every referenced artifact, recomputes both
independent validator reports, and checks the v2 promotion receipt binding. It
does not accept prompts, responses, tokens, or credentials in the bundle.
