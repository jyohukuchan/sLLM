FROM ghcr.io/open-webui/open-webui@sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff

LABEL org.opencontainers.image.title="uLLM OpenWebUI" \
      org.opencontainers.image.description="OpenWebUI 0.9.4 with fail-closed provider stream error handling" \
      org.opencontainers.image.version="0.9.4-ullm.1" \
      org.opencontainers.image.source="https://github.com/jyohukuchan/uLLM-project" \
      org.opencontainers.image.base.name="ghcr.io/open-webui/open-webui" \
      org.opencontainers.image.base.digest="sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff" \
      io.ullm.openwebui.base.image.id="sha256:18247c4608796dd5e416ec1e82f20457837a219ed9c272a8d64b405a262b3399" \
      io.ullm.openwebui.patch.sha256="20bf654b96f005d5008deff0cdd6f9cd62cbd21fe20a3680b66aecb36190813a" \
      io.ullm.openwebui.middleware.base.sha256="6246723f9ae6dcf99407c706e325c4bfac285cc821fbef9cb4a7694c6c39e204" \
      io.ullm.openwebui.middleware.sha256="b8aa5524fac6971aa8326cbef024b6fe9bcea03b3a00d4e7b0fa559514e0c66a"

COPY provider-stream-error.patch /tmp/provider-stream-error.patch

RUN set -eux; \
    middleware=/app/backend/open_webui/utils/middleware.py; \
    echo "6246723f9ae6dcf99407c706e325c4bfac285cc821fbef9cb4a7694c6c39e204  ${middleware}" | sha256sum -c -; \
    patch --batch --forward --fuzz=0 -p1 -d /app/backend < /tmp/provider-stream-error.patch; \
    echo "b8aa5524fac6971aa8326cbef024b6fe9bcea03b3a00d4e7b0fa559514e0c66a  ${middleware}" | sha256sum -c -; \
    PYTHONPYCACHEPREFIX=/tmp/openwebui-pycache python -m py_compile "${middleware}"; \
    rm -rf /tmp/openwebui-pycache /tmp/provider-stream-error.patch
