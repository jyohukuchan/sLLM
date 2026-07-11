# uLLM OpenWebUI deployment

This deployment runs exactly one resident SQ8 worker behind the local
OpenAI-compatible gateway. Port 8000 is bound to the fixed
`open-webui-network` gateway and is dropped by nftables on every other input
interface.

## Fixed identities

- gateway address: `http://172.20.0.1:8000/v1`
- Docker network: `open-webui-network` (`172.20.0.0/16`)
- bridge interface: `br-79bb7cfca31c`
- model: `ullm-qwen3-14b-sq8`
- OpenWebUI: `http://192.168.0.66:3000`

The OpenWebUI image is pinned by digest in `openwebui/compose.yaml`. Existing
connections and the external `open-webui` volume are preserved. Its session
signing key is mounted read-only from `/etc/ullm/openwebui-secret-key`, so a
container replacement does not invalidate every login session.

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
uLLM provider/model, records the 4096-token model context as metadata, and
disables title, follow-up, and tag background generation. OpenWebUI v0.9.4
does not enforce a context length for OpenAI-compatible providers. It sends the
complete selected history, and the uLLM gateway remains the authoritative 4096
token limit with a visible HTTP 400 on overflow. Do not add `num_ctx`: this
OpenWebUI version forwards it to OpenAI-compatible providers as an unsupported
field.

```bash
cd /home/homelab1/coding-local/ultimateLLM/uLLM-project
docker stop open-webui 2>/dev/null || true
docker run --rm \
  -v open-webui:/data \
  -v /etc/ullm/openai-api-key:/run/secrets/ullm-api-key:ro \
  -v "$PWD/deploy/openwebui/configure.py:/configure.py:ro" \
  --entrypoint python \
  ghcr.io/open-webui/open-webui@sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff \
  /configure.py
docker compose -f deploy/openwebui/compose.yaml up -d
```

OpenWebUI is ready when both commands succeed:

```bash
curl --fail --silent http://127.0.0.1:3000/health
docker inspect --format '{{.State.Health.Status}}' open-webui
```

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

`docker compose down` does not remove the external OpenWebUI volume. Never add
`--volumes` during routine recovery. Stopping `ullm-openai-firewall.service`
removes only the dedicated `inet ullm_openai` table and also stops the gateway
through the systemd dependency.

For an upgrade, first update and verify the worker, gateway lockfile, and pinned
OpenWebUI digest. Re-run `configure.py`, restart both services, and retain the
new backup under the OpenWebUI volume until the smoke matrix passes.
