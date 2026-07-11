# uLLM OpenAI gateway

This package is the single-process OpenAI Chat Completions gateway for the
resident `ullm-sq8-worker`. P8-D supports non-streaming requests only. A
`stream=true` request is rejected instead of being silently converted to a
non-streaming request; SSE is added separately in P8-E.

## Runtime contract

- Python 3.12
- one Uvicorn process and one resident Rust worker
- one active request and no waiting queue
- frozen local Qwen3 tokenizer with network access disabled
- fixed public model ID `ullm-qwen3-14b-sq8`
- Bearer authentication on both `/v1` endpoints
- fail-closed worker identity, JSONL ordering, singleton lock, and watchdogs

## Install

```bash
cd services/openai-gateway
uv sync --frozen --no-dev
```

The lock file pins every direct and transitive dependency. Do not install this
package into the vLLM development environment.

## Local smoke

Create a one-line API key file with mode `0600`, then set the deployment paths:

```bash
export ULLM_WORKER_BINARY="$PWD/../../target/release/ullm-sq8-worker"
export ULLM_ARTIFACT_DIR=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact
export ULLM_PACKAGE_DIR=/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package
export ULLM_TOKENIZER_DIR=/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8
export ULLM_API_KEY_FILE=/path/to/one-line-key
export ULLM_GPU_LOCK_FILE=/tmp/ullm-r9700.lock
export ULLM_BIND_HOST=127.0.0.1
uv run --frozen ullm-openai-gateway
```

The product service will use the fixed Docker bridge address and a lock under
`/run/lock`; loopback and `/tmp` above are only for a local smoke.

After `/readyz` returns 200:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $(<"$ULLM_API_KEY_FILE")" \
  -H 'Content-Type: application/json' \
  -d '{"model":"ullm-qwen3-14b-sq8","messages":[{"role":"user","content":"日本語で短く挨拶してください。"}],"stream":false,"max_tokens":32}'
```

## Verification

```bash
uv run --frozen pytest -q
uv run --frozen mypy --strict src/ullm_openai_gateway
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
```
