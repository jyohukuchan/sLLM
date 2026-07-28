#!/usr/bin/env bash
# Root-only lifecycle wrapper. It deliberately releases the R9700 flock in
# the user runner before starting ullm-openai.
set -euo pipefail
repo=/home/homelab1/coding-local/ultimateLLM/uLLM-project
out="$repo/benchmarks/results/2026-07-28/gemma4-prefill-definitive-eh-v1.0"
mkdir -p "$out/service"
restore_service() {
    systemctl start ullm-openai.service > "$out/service/start.stdout" 2> "$out/service/start.stderr" || true
    systemctl show ullm-openai.service -p ActiveState -p NRestarts > "$out/service/post-start.txt" || true
}
trap restore_service EXIT
systemctl show ullm-openai.service -p ActiveState -p NRestarts > "$out/service/pre-stop.txt"
systemctl stop ullm-openai.service > "$out/service/stop.stdout" 2> "$out/service/stop.stderr"
systemctl show ullm-openai.service -p ActiveState -p NRestarts > "$out/service/post-stop.txt"
runner="${ULLM_ROOT_WRAPPER_RUNNER:-$out/run-measurement.sh}"
case "$runner" in
    "$repo"/*) ;;
    *)
        echo "ULLM_ROOT_WRAPPER_RUNNER must name a script below $repo" >&2
        exit 2
        ;;
esac
runuser -u homelab1 -- env HOME=/home/homelab1 ULLM_EH_RESUME="${ULLM_EH_RESUME:-0}" PATH=/home/homelab1/.cargo/bin:/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin \
    bash "$runner"
