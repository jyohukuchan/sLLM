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

The runner pins `HIP_VISIBLE_DEVICES=1`, matching the deployed WRX80 isolation,
for both the resident v2 worker and the sequential legacy comparison. It does
not stop or restart the active services. Before starting either process, it
also runs `rocm-smi --showpids --json` and fails closed if the target R9700 has
positive-VRAM processes, so the operator must provide the exclusive window
first.

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

### v2 OpenWebUI gates

Run these gates only after the v2 manifest has been activated and OpenWebUI has
been reconciled from that manifest. They must run during an exclusive R9700
window. The model overrides are required: without them, the shared gate tools
default to the legacy SQ8 model and the evidence would be for the wrong
deployment.

Use immutable local Docker content identities for the browser and probe images.
The example identities below are the images currently present on WRX80; obtain
fresh identities with `docker image inspect` before a later run.

```bash
export ULLM_MODEL_ID=ullm-qwen3.5-9b-aq4
export ULLM_MODEL_NAME='uLLM Qwen3.5 9B AQ4'
export TOKEN_FILE=/etc/ullm/openai-api-key
export BROWSER_IMAGE=sha256:dbd552f6c831816050a1381a54cdb8d37df56df7f6559c82aba451d2ea93e0aa
export PROBE_IMAGE=sha256:5dce198cca467ce79994ed65e01d03882238f9efdd16a8c6f4bc55151c8a4a54
export OPENWEBUI_IMAGE=ullm/open-webui@sha256:ef5ae4fbc06abb662eeefe87e584ea7c69e55838f5f08f637057b9108048b409
export OPENWEBUI_URL=http://192.168.0.66:3000/
export SERVICE=ullm-openai.service
export OUT=benchmarks/results/2026-07-13/qwen35-9b-aq4-reasoning-v0.1
# Set PROMOTION_SOURCE_COMMIT to the 40-character commit in the v2 receipt.
mkdir -p "$OUT"
```

Collect the five hash-only HTTP/SSE release cases after the v2 manifest is
active. The collector validates the manifest first, then requires an exclusive
gfx1201/R9700 worker and rejects resident `llama-server` or other GPU owners.
It observes only sanitized `request_released` lifecycle events and never
publishes prompts, responses, request bodies, or credentials. The output is
atomic and contains `cases.json`, `lifecycle.json`, `resource-samples.jsonl`,
and a bounded `summary.json`.

```bash
uv run --project services/openai-gateway python \
  tools/run-generic-reasoning-release-campaign.py \
  --output-dir "$OUT/http-sse-campaign" \
  --manifest /etc/ullm/served-models/active.json \
  --fixture-suite tests/fixtures/generic-reasoning-release-v0.1/prompts.json \
  --token-file "$TOKEN_FILE" --http-image "$PROBE_IMAGE" \
  --service "$SERVICE"

uv run --project services/openai-gateway python \
  tools/prepare-generic-reasoning-release-evidence.py \
  --cases "$OUT/http-sse-campaign/cases.json" \
  --lifecycle "$OUT/http-sse-campaign/lifecycle.json" \
  --manifest /etc/ullm/served-models/active.json \
  --worker-binary target/reasoning-v2/release/ullm-aq4-worker \
  --openwebui-image "$OPENWEBUI_IMAGE" \
  --active-promotion-source-commit "$PROMOTION_SOURCE_COMMIT" \
  --output "$OUT/release-evidence.json" --status incomplete
```

The campaign must be rerun if the active manifest, worker binary, promotion
receipt, tokenizer, or source identity changes. Do not run it while the
legacy llama.cpp comparison service owns the target R9700.

The image identities above are the current local content identities; they MUST
be refreshed with `docker image inspect` before a later run. Run the normal
100-chat soak, the restart-recovery
20-chat soak, Stop, and worker-failure gates as separate output directories:

```bash
ULLM_OPENWEBUI_SOAK_COUNT=100 uv run --project services/openai-gateway python \
  tools/run-openwebui-soak-gate.py \
  --output-dir "$OUT/soak-100" --token-file "$TOKEN_FILE" \
  --browser-image "$BROWSER_IMAGE" --openwebui-url "$OPENWEBUI_URL" \
  --service "$SERVICE" --include-smoke

systemctl restart "$SERVICE"
ULLM_OPENWEBUI_SOAK_COUNT=20 uv run --project services/openai-gateway python \
  tools/run-openwebui-soak-gate.py \
  --output-dir "$OUT/soak-restart-20" --token-file "$TOKEN_FILE" \
  --browser-image "$BROWSER_IMAGE" --openwebui-url "$OPENWEBUI_URL" \
  --service "$SERVICE" --include-smoke

uv run --project services/openai-gateway python tools/run-openwebui-stop-gate.py \
  --output-dir "$OUT/stop" --token-file "$TOKEN_FILE" \
  --browser-image "$BROWSER_IMAGE" --openwebui-url "$OPENWEBUI_URL" \
  --service "$SERVICE"

uv run --project services/openai-gateway python tools/run-openwebui-failure-gate.py \
  --output-dir "$OUT/failure" --token-file "$TOKEN_FILE" \
  --browser-image "$BROWSER_IMAGE" --probe-image "$PROBE_IMAGE" \
  --openwebui-url "$OPENWEBUI_URL" --service "$SERVICE"
```

Run the v2-specific browser smoke separately through its safe runner. It binds
the candidate and switch model IDs into the container environment, checks the
four observed provider-request hashes against those IDs, validates the v2
schema, and atomically publishes only gate-eligible hash-only evidence:

```bash
uv run --project services/openai-gateway python \
  tools/run-openwebui-reasoning-browser-smoke.py \
  --output "$OUT/browser-reasoning.json" \
  --manifest /etc/ullm/served-models/active.json \
  --token-file "$TOKEN_FILE" \
  --browser-image "$BROWSER_IMAGE" --openwebui-url "$OPENWEBUI_URL" \
  --model-id "$ULLM_MODEL_ID" --model-name "$ULLM_MODEL_NAME" \
  --switch-model-id llama-qwen3.5-9b-ud-q4 \
  --switch-model-name 'llama.cpp Qwen3.5 9B UD-Q4_K_XL'

uv run --project services/openai-gateway python \
  tools/validate-openwebui-reasoning-browser-smoke.py \
  "$OUT/browser-reasoning.json" --require-pass > "$OUT/browser-reasoning-validator.json"
```

Never run these gates against the active v1 service and label their output as
v2 evidence.

For the final hash-only handoff, keep the generic release evidence, its
validator report, the OpenWebUI browser evidence and report, and the promotion
evidence and receipt beside a `ullm.generic_reasoning_release_bundle.v1`
document. Stage those six artifacts and the rollback source files under one
bundle directory, then assemble and validate the bundle with:

Measured case records can be assembled with
`tools/prepare-generic-reasoning-release-evidence.py`; it computes the current
Git/worktree and artifact identities and runs the release validator before
publishing. Use `--status incomplete` during collection and reserve
`--status complete` for the final aligned, clean, gate-eligible run.
Pass `--lifecycle` with sanitized `request_released` records when assembling a
complete artifact; the validator matches those records to every measured case.

`tools/prepare-generic-reasoning-release-bundle.py` hashes the staged artifacts
and the previous active manifest, systemd unit, and environment file. It writes
only relative, non-symlink references and invokes the bundle validator before
publishing. The six artifact paths must be below the output bundle directory.

```bash
uv run --project services/openai-gateway python \
  tools/prepare-generic-reasoning-release-bundle.py \
  --release-evidence /path/to/release.json \
  --release-validator /path/to/release-validator.json \
  --browser-evidence /path/to/browser.json \
  --browser-validator /path/to/browser-validator.json \
  --promotion-evidence /path/to/promotion-evidence.json \
  --promotion-receipt /path/to/promotion-receipt.json \
  --rollback-manifest /path/to/previous-active.json \
  --systemd-unit /path/to/ullm-openai.service \
  --environment-file /path/to/ullm-openai.env \
  --output /path/to/release-bundle.json \
  --status incomplete
```

```bash
uv run --project services/openai-gateway python \
  tools/validate-generic-reasoning-release-bundle.py \
  /path/to/generic-reasoning-release-bundle.json \
  --require-complete
```

The bundle validator re-hashes every referenced artifact, recomputes both
independent validator reports, and checks the v2 promotion receipt binding. It
does not accept prompts, responses, tokens, or credentials in the bundle.
