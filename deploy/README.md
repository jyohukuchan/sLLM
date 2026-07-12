# uLLM OpenWebUI deployment

This deployment runs exactly one resident worker behind the local
OpenAI-compatible gateway. SQ8 remains the default, while the worker and model
metadata can be switched to AQ4 without changing the OpenWebUI configuration
tool. Port 8000 is bound to the fixed
`open-webui-network` gateway and is dropped by nftables on every other input
interface.

## Default identities

- gateway address: `http://172.20.0.1:8000/v1`
- Docker network: `open-webui-network` (`172.20.0.0/16`)
- bridge interface: `br-79bb7cfca31c`
- model: `ullm-qwen3-14b-sq8`
- OpenWebUI: `http://192.168.0.66:3000`

`openwebui/Dockerfile` pins the OpenWebUI 0.9.4 base image to digest
`sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff`
and builds the local image `ullm/open-webui:0.9.4-ullm.1`. The build rejects a
base middleware file whose SHA256 is not the expected value, applies the local
provider-stream-error patch with zero fuzz, checks the fixed post-patch SHA256,
and compiles the result. Existing connections and the external `open-webui`
volume are preserved. Its session signing key is mounted read-only from
`/etc/ullm/openwebui-secret-key`, so a container replacement does not invalidate
every login session. Automatic version checks are disabled because this
deployment upgrades only after verifying a new pinned image digest and local
patch hashes; this also prevents an update notification from covering the chat
Stop control.

## Install the gateway

Prepare the locked Python environment before installing the service:

```bash
cd /home/homelab1/coding-local/ultimateLLM/uLLM-project/services/openai-gateway
uv sync --frozen --no-dev --offline
```

Install the root-managed configuration. Generate the API key only when the
file is absent; replacing it requires re-running the OpenWebUI configuration
step below.

```bash
sudo install -d -m 0750 -o root -g homelab1 /etc/ullm
sudo install -m 0644 ../../deploy/systemd/ullm-openai.env.example /etc/ullm/openai-gateway.env
if ! sudo test -f /etc/ullm/openai-api-key; then
  openssl rand -hex 32 | sudo tee /etc/ullm/openai-api-key >/dev/null
fi
sudo chown root:homelab1 /etc/ullm/openai-api-key
sudo chmod 0640 /etc/ullm/openai-api-key
if ! sudo test -f /etc/ullm/openwebui-secret-key; then
  openssl rand -hex 32 | sudo tee /etc/ullm/openwebui-secret-key >/dev/null
fi
sudo chown root:root /etc/ullm/openwebui-secret-key
sudo chmod 0600 /etc/ullm/openwebui-secret-key
sudo install -m 0644 ../../deploy/nftables/ullm-openai.nft /etc/ullm/ullm-openai.nft
sudo install -m 0755 ../../deploy/nftables/ullm-openai-firewall /usr/local/libexec/ullm-openai-firewall
sudo install -m 0644 ../../deploy/systemd/ullm-openai-firewall.service /etc/systemd/system/
sudo install -m 0644 ../../deploy/systemd/ullm-openai.service /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/ullm-openai-firewall.service /etc/systemd/system/ullm-openai.service
sudo systemctl daemon-reload
sudo systemctl enable --now ullm-openai.service
```

The installed service above remains the legacy environment deployment. Do not
install the manifest-mode drop-in until the selected release worker accepts
`--served-model-manifest` and a real, validated candidate manifest has been
generated. This keeps the currently deployed AQ4 compatibility service
unchanged while manifest mode is prepared.

### Enable manifest mode

Manifest mode uses `/etc/ullm/served-models/active.json` as the single active
model contract. It is an atomically replaced regular file, not a symlink.
Candidate manifests remain separately named files under
`/etc/ullm/served-models/candidates/`. Install the operations-only environment
and the optional systemd drop-in as follows, but do not restart the service
until the first active manifest has passed validation:

```bash
sudo install -d -m 0750 -o root -g homelab1 /etc/ullm/served-models
sudo install -d -m 0750 -o root -g homelab1 /etc/ullm/served-models/candidates
sudo install -m 0644 deploy/systemd/ullm-openai-manifest.env.example \
  /etc/ullm/openai-gateway-manifest.env
sudo install -d -m 0755 /etc/systemd/system/ullm-openai.service.d
sudo install -m 0644 \
  deploy/systemd/ullm-openai.service.d/10-served-model.conf \
  /etc/systemd/system/ullm-openai.service.d/10-served-model.conf
sudo systemd-analyze verify ullm-openai.service
sudo systemctl daemon-reload
```

The drop-in clears the base unit's legacy `EnvironmentFile` before loading
`/etc/ullm/openai-gateway-manifest.env`. Its `ExecStartPre` runs
`tools/validate-served-model.py` against the active manifest. A missing,
modified, unsafe, or identity-mismatched manifest therefore prevents the
gateway from starting. The gateway validates the same document again before
launching the worker.

Manifest mode and legacy model variables are mutually exclusive. In
particular, never place `ULLM_WORKER_BINARY`, `ULLM_PRODUCT_ROOT`,
`ULLM_TOKENIZER_DIR`, `ULLM_MODEL_ID`, `ULLM_MODEL_NAME`,
`ULLM_MODEL_DESCRIPTION`, model limits, hashes, worker arguments, tokenizer
profiles, or `ULLM_HIP_GUARDS` in
`/etc/ullm/openai-gateway-manifest.env`. The gateway rejects the process if any
legacy model variable is present with `ULLM_SERVED_MODEL_MANIFEST`; it does not
choose one source by precedence.

The gateway remains unready while the model is loading. Check readiness from
the Docker network because the host firewall intentionally rejects loopback
and LAN access to the bridge address.

```bash
docker run --rm --network open-webui-network \
  -v /etc/ullm/openai-api-key:/run/secrets/ullm-api-key:ro \
  --entrypoint sh \
  ghcr.io/open-webui/open-webui@sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff \
  -c 'curl --fail --silent --show-error -H "Authorization: Bearer $(cat /run/secrets/ullm-api-key)" http://172.20.0.1:8000/v1/models'
```

## Configure and start OpenWebUI

Stop OpenWebUI before editing its SQLite database. The configuration tool uses
SQLite's backup API, retains every existing provider, adds or updates only the
selected uLLM provider/model, records the selected model context as metadata, and
enables terminal usage collection while disabling title, follow-up, and tag
background generation. OpenWebUI then requests the final usage chunk and merges
the gateway's llama-server-compatible `timings` into the response information
shown below each assistant message. That information includes
`predicted_per_second`, `finish_reason`, and `termination_reason`. OpenWebUI
v0.9.4 does not enforce a context length for OpenAI-compatible providers. It
sends the complete selected history, and the uLLM gateway remains the
authoritative 4096-token limit with a visible HTTP 400 on overflow. Do not add
`num_ctx`: this OpenWebUI version forwards it to OpenAI-compatible providers as
an unsupported field.

```bash
cd /home/homelab1/coding-local/ultimateLLM/uLLM-project
docker compose -f deploy/openwebui/compose.yaml build open-webui
deploy/openwebui/verify-derived-image.sh
docker stop open-webui 2>/dev/null || true
docker run --rm \
  --env-file /etc/ullm/openai-gateway.env \
  -v open-webui:/data \
  -v /etc/ullm/openai-api-key:/run/secrets/ullm-api-key:ro \
  -v "$PWD/deploy/openwebui/configure.py:/configure.py:ro" \
  --entrypoint python \
  ullm/open-webui:0.9.4-ullm.1 \
  /configure.py
docker compose -f deploy/openwebui/compose.yaml up -d --no-build
```

In manifest mode, mount the active-manifest directory read-only and pass the
manifest explicitly. Do not also pass the four legacy model metadata options
or their environment variables:

```bash
docker stop open-webui 2>/dev/null || true
docker run --rm \
  -v open-webui:/data \
  -v /etc/ullm/openai-api-key:/run/secrets/ullm-api-key:ro \
  -v /etc/ullm/served-models:/etc/ullm/served-models:ro \
  -v "$PWD/deploy/openwebui/configure.py:/configure.py:ro" \
  --entrypoint python \
  ullm/open-webui:0.9.4-ullm.1 \
  /configure.py \
  --served-model-manifest /etc/ullm/served-models/active.json \
  --previous-managed-model-id ullm-qwen3-14b-sq8 \
  --base-url http://172.20.0.1:8000/v1
docker compose -f deploy/openwebui/compose.yaml up -d --no-build
```

The previous-model argument is explicit migration authority for a uLLM model
row created before `meta.ullm` management markers existed. It may be repeated,
and only those unmarked IDs are marked as uLLM-managed and made inactive. A
safer identity-preserving alternative is
`--previous-served-model-manifest /etc/ullm/served-models/candidates/qwen3-14b-sq8.json`,
which also records the previous manifest digest. Existing marked uLLM rows at
the same base URL continue to be reconciled automatically. No model-name or ID
prefix inference is performed, and unrelated model/provider rows are unchanged.
Both previous-model options require manifest mode.

For non-interactive invocation, `ULLM_PREVIOUS_MANAGED_MODEL_IDS` is a JSON
string array and `ULLM_PREVIOUS_SERVED_MODEL_MANIFEST` is one manifest path.
They correspond to the repeatable command-line options. Duplicate previous IDs,
including an ID supplied once directly and once through a manifest, are rejected.

`configure.py` reads the following variables from the gateway environment file.
Every value also has a matching command-line option, and an explicit option
takes precedence over the environment:

| Environment variable | Command-line option | SQ8 default |
| --- | --- | --- |
| `ULLM_OPENAI_BASE_URL` | `--base-url` | `http://172.20.0.1:8000/v1` |
| `ULLM_MODEL_ID` | `--model-id` | `ullm-qwen3-14b-sq8` |
| `ULLM_MODEL_NAME` | `--model-name` | `uLLM Qwen3 14B SQ8` |
| `ULLM_MODEL_CONTEXT_LENGTH` | `--context-length` | `4096` |
| `ULLM_MODEL_DESCRIPTION` | `--description` | `Qwen3 14B served locally by uLLM SQ8_0.` |

For Qwen3.5 9B AQ4, set the worker, product root, and tokenizer paths for that
runtime, then use values such as the following in
`/etc/ullm/openai-gateway.env`:

```ini
ULLM_WORKER_BINARY=/home/homelab1/coding-local/ultimateLLM/uLLM-project/target/release/ullm-aq4-worker
ULLM_PRODUCT_ROOT=/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1
ULLM_TOKENIZER_DIR=/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B
ULLM_TOKENIZER_PROFILE=qwen35-9b
ULLM_MODEL_ID=ullm-qwen3.5-9b-aq4
ULLM_MODEL_NAME=uLLM Qwen3.5 9B AQ4
ULLM_MODEL_CONTEXT_LENGTH=4096
ULLM_MODEL_DESCRIPTION=Qwen3.5 9B served locally by uLLM AQ4_0.
ULLM_MODEL_REVISION=aq4-cli-compat-v0.1
ULLM_ARTIFACT_CONTENT_SHA256=a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad
ULLM_PACKAGE_MANIFEST_SHA256=a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad
ULLM_DEVICE=gfx1201
ULLM_EXECUTION_PROFILE=rdna4_aq4_cli_compat
ULLM_MAX_NEW_TOKENS=512
ULLM_VOCAB_SIZE=248320
ULLM_EOS_TOKEN_IDS=248044,248046
ULLM_TOP_K=1
ULLM_HIP_VISIBLE_DEVICES=1
ULLM_HIP_GUARDS=
ULLM_OPENAI_BASE_URL=http://172.20.0.1:8000/v1
```

This block is a legacy-mode example only. None of these model profile values
may be copied into the manifest-mode environment file.

Restart the gateway before re-running the configuration command. Reusing the
same base URL switches the single resident backend while retaining the prior
SQ8 model row in OpenWebUI. A different base URL adds another provider. The
gateway model ID and `configure.py` model ID must always match; otherwise
OpenWebUI will select a model that the endpoint rejects. Only one worker needs
to be resident when the workers share the same GPU lock.

OpenWebUI is ready when both commands succeed:

```bash
curl --fail --silent http://127.0.0.1:3000/health
docker inspect --format '{{.State.Health.Status}}' open-webui
```

## Known v0.1 limitations

- The product runs one active GPU request with no waiting queue or request batching. A concurrent request receives `429` with `Retry-After: 1`; OpenWebUI v0.9.4 may present this as a visible HTTP 400 busy error.
- The context limit is 4096 tokens and the gateway rejects overflow without truncating chat history.
- The API is text-only Chat Completions for one loaded model at a time. Tools, structured output guarantees, multimodal input, embeddings, and the Responses API are not supported.
- Request stop strings and automatic whole-turn history truncation are not implemented. Model EOS, maximum-token completion, and the OpenWebUI Stop action are supported.
- AQ4 legacy mode remains available as a transition and rollback path and reloads the package for every request. Manifest mode uses the resident AQ4 session and streams token events live; production deployments should use manifest mode.
- TLS termination and multi-tenant authorization are outside this local bridge-only deployment.

## Operations

```bash
sudo systemctl status ullm-openai.service
sudo journalctl -u ullm-openai.service -f
sudo systemctl restart ullm-openai.service
sudo systemctl stop ullm-openai.service
sudo systemctl start ullm-openai.service
docker compose -f deploy/openwebui/compose.yaml logs -f --tail=100
docker compose -f deploy/openwebui/compose.yaml restart
docker compose -f deploy/openwebui/compose.yaml down
```

### Atomic manifest activation and rollback

`tools/activate-served-model.py` validates and copies a candidate, atomically
replaces `/etc/ullm/served-models/active.json`, and restores the prior bytes if
any later command fails. Commands are JSON arrays executed directly without a
shell. The tool supplies `ULLM_ACTIVE_MANIFEST`,
`ULLM_ACTIVE_MANIFEST_SHA256`, `ULLM_ACTIVE_MODEL_ID`, and
`ULLM_ACTIVATION_STAGE` to each hook. Arguments are not shell-expanded, so the
examples use the fixed active path where a child process needs it.

The following example restarts the gateway, waits for its unauthenticated
bridge-only readiness endpoint, reconciles OpenWebUI from the same active
manifest, and verifies the UI health endpoint. `curl` retry behavior is inside
the fixed container command; no shell wrapper is involved.

```bash
sudo python3 tools/activate-served-model.py \
  --candidate /etc/ullm/served-models/candidates/qwen35-9b-aq4.json \
  --active-manifest /etc/ullm/served-models/active.json \
  --command-timeout-seconds 650 \
  --check-command-json '["/usr/bin/systemctl","restart","ullm-openai.service"]' \
  --check-command-json '["/usr/bin/docker","run","--rm","--network","open-webui-network","--entrypoint","curl","ghcr.io/open-webui/open-webui@sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff","--fail","--silent","--show-error","--retry","120","--retry-delay","2","--retry-max-time","600","--retry-all-errors","http://172.20.0.1:8000/readyz"]' \
  --reconcile-command-json '["/usr/bin/docker","stop","open-webui"]' \
  --reconcile-command-json '["/usr/bin/docker","run","--rm","-v","open-webui:/data","-v","/etc/ullm/openai-api-key:/run/secrets/ullm-api-key:ro","-v","/etc/ullm/served-models:/etc/ullm/served-models:ro","-v","/home/homelab1/coding-local/ultimateLLM/uLLM-project/deploy/openwebui/configure.py:/configure.py:ro","--entrypoint","python","ullm/open-webui:0.9.4-ullm.1","/configure.py","--served-model-manifest","/etc/ullm/served-models/active.json","--previous-served-model-manifest","/etc/ullm/served-models/candidates/qwen3-14b-sq8.json","--base-url","http://172.20.0.1:8000/v1"]' \
  --reconcile-command-json '["/usr/bin/docker","compose","-f","/home/homelab1/coding-local/ultimateLLM/uLLM-project/deploy/openwebui/compose.yaml","up","-d","--no-build"]' \
  --final-check-command-json '["/usr/bin/curl","--fail","--silent","--show-error","--retry","30","--retry-delay","2","--retry-connrefused","http://127.0.0.1:3000/health"]' \
  --rollback-command-json '["/usr/bin/systemctl","restart","ullm-openai.service"]' \
  --rollback-command-json '["/usr/bin/docker","run","--rm","--network","open-webui-network","--entrypoint","curl","ghcr.io/open-webui/open-webui@sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff","--fail","--silent","--show-error","--retry","120","--retry-delay","2","--retry-max-time","600","--retry-all-errors","http://172.20.0.1:8000/readyz"]' \
  --rollback-command-json '["/usr/bin/docker","stop","open-webui"]' \
  --rollback-command-json '["/usr/bin/docker","run","--rm","-v","open-webui:/data","-v","/etc/ullm/openai-api-key:/run/secrets/ullm-api-key:ro","-v","/etc/ullm/served-models:/etc/ullm/served-models:ro","-v","/home/homelab1/coding-local/ultimateLLM/uLLM-project/deploy/openwebui/configure.py:/configure.py:ro","--entrypoint","python","ullm/open-webui:0.9.4-ullm.1","/configure.py","--served-model-manifest","/etc/ullm/served-models/active.json","--previous-served-model-manifest","/etc/ullm/served-models/candidates/qwen35-9b-aq4.json","--base-url","http://172.20.0.1:8000/v1"]' \
  --rollback-command-json '["/usr/bin/docker","compose","-f","/home/homelab1/coding-local/ultimateLLM/uLLM-project/deploy/openwebui/compose.yaml","up","-d","--no-build"]'
```

Rollback hooks run only after the old active-manifest bytes have been restored.
They must therefore restart the gateway and reconcile OpenWebUI again from the
fixed active path. Hook stdout and stderr are intentionally discarded. If a
rollback hook fails, the tool reports `activation and rollback failed`; inspect
the systemd and Docker journals before attempting another activation.
The example is specifically an SQ8-to-AQ4 activation: its reconcile hook names
the old SQ8 candidate manifest, while its rollback hook retires the attempted
AQ4 candidate. Update both previous-manifest paths for a different transition.

`docker compose down` does not remove the external OpenWebUI volume. Never add
`--volumes` during routine recovery. Stopping `ullm-openai-firewall.service`
removes only the dedicated `inet ullm_openai` table and also stops the gateway
through the systemd dependency.

For an upgrade, first update and verify the worker, gateway lockfile, pinned
OpenWebUI digest, middleware input/output hashes, and zero-fuzz patch. Rebuild
and run `verify-derived-image.sh` before replacing the container. Re-run
`configure.py`, restart both services, and retain the new backup under the
OpenWebUI volume until the smoke matrix passes. Do not use `docker compose pull`
for the local derived image.
