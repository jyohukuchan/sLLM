#!/bin/sh
set -eu

base_image='ghcr.io/open-webui/open-webui@sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff'
image="${1:-ullm/open-webui:0.9.4-ullm.1}"
base_middleware_sha256='6246723f9ae6dcf99407c706e325c4bfac285cc821fbef9cb4a7694c6c39e204'
middleware_sha256='b8aa5524fac6971aa8326cbef024b6fe9bcea03b3a00d4e7b0fa559514e0c66a'
base_image_id='sha256:18247c4608796dd5e416ec1e82f20457837a219ed9c272a8d64b405a262b3399'
patch_sha256='20bf654b96f005d5008deff0cdd6f9cd62cbd21fe20a3680b66aecb36190813a'
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)

require_label() {
    label=$1
    expected=$2
    actual=$(docker image inspect --format "{{ index .Config.Labels \"${label}\" }}" "${image}")
    if [ "${actual}" != "${expected}" ]; then
        printf 'label mismatch: %s\nexpected: %s\nactual: %s\n' "${label}" "${expected}" "${actual}" >&2
        exit 1
    fi
}

work_dir=$(mktemp -d)
trap 'rm -rf "${work_dir}"' EXIT HUP INT TERM
actual_base_image_id=$(docker image inspect --format '{{.Id}}' "${base_image}")
if [ "${actual_base_image_id}" != "${base_image_id}" ]; then
    printf 'base image ID mismatch\nexpected: %s\nactual: %s\n' \
        "${base_image_id}" "${actual_base_image_id}" >&2
    exit 1
fi
echo "${patch_sha256}  ${script_dir}/provider-stream-error.patch" | sha256sum -c -
docker image inspect --format '{{range .RootFS.Layers}}{{println .}}{{end}}' "${base_image}" \
    | sed '/^$/d' > "${work_dir}/base-layers"
docker image inspect --format '{{range .RootFS.Layers}}{{println .}}{{end}}' "${image}" \
    | sed '/^$/d' > "${work_dir}/image-layers"
base_layer_count=$(wc -l < "${work_dir}/base-layers")
head -n "${base_layer_count}" "${work_dir}/image-layers" > "${work_dir}/image-base-layers"
cmp "${work_dir}/base-layers" "${work_dir}/image-base-layers"
if [ "$(wc -l < "${work_dir}/image-layers")" -le "${base_layer_count}" ]; then
    echo 'derived image has no layer after the pinned base image' >&2
    exit 1
fi

require_label org.opencontainers.image.title 'uLLM OpenWebUI'
require_label org.opencontainers.image.version '0.9.4-ullm.1'
require_label org.opencontainers.image.source 'https://github.com/jyohukuchan/uLLM-project'
require_label org.opencontainers.image.base.name 'ghcr.io/open-webui/open-webui'
require_label org.opencontainers.image.base.digest 'sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff'
require_label io.ullm.openwebui.base.image.id "${base_image_id}"
require_label io.ullm.openwebui.patch.sha256 "${patch_sha256}"
require_label io.ullm.openwebui.middleware.base.sha256 "${base_middleware_sha256}"
require_label io.ullm.openwebui.middleware.sha256 "${middleware_sha256}"

docker run --rm --entrypoint sh "${image}" -ec "
middleware=/app/backend/open_webui/utils/middleware.py
echo '${middleware_sha256}  /app/backend/open_webui/utils/middleware.py' | sha256sum -c -
PYTHONPYCACHEPREFIX=/tmp/openwebui-pycache python -m py_compile \"\${middleware}\"
"

printf 'verified derived OpenWebUI image: %s\n' "${image}"
