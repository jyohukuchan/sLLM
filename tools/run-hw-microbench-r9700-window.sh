#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
# One exclusive R9700 window for the standalone hardware microbenchmark.
# It preserves active.json and always releases the GPU lock before restart.
set -Euo pipefail
[[ $# == 1 && $1 == /* ]] || { echo "usage: $0 /absolute/results-dir" >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
out=$(realpath -m "$1")
service=ullm-openai.service
lock=/run/ullm/r9700.lock
manifest=/etc/ullm/served-models/active.json
[[ ! -e $out ]] || { echo "refusing to reuse results directory: $out" >&2; exit 2; }
mkdir -p "$out"/{preflight,service,telemetry}
sudo_systemctl() { printf '%s\n' Threadripper | sudo -S -p '' systemctl "$@"; }
metric() { amd-smi metric --gpu 2 --temperature --clock --power --violation --json; }
state() {
  local name=$1
  fuser -v "$lock" >"$out/preflight/$name-fuser.txt" 2>&1 || true
  pgrep -af 'codex exec' >"$out/preflight/$name-codex-exec.txt" || true
  systemctl show "$service" -p ActiveState -p NRestarts >"$out/preflight/$name-service.txt"
}
stopped=0
held=0
release_lock() { if (( held )); then flock -u 9 || true; exec 9>&-; held=0; fi; }
restore() {
  local status=$?
  trap - EXIT
  release_lock
  if (( stopped )); then
    sudo_systemctl start "$service" >"$out/service/start.stdout" 2>"$out/service/start.stderr" || status=1
    for _ in $(seq 1 45); do systemctl is-active --quiet "$service" && break; sleep 1; done
  fi
  systemctl show "$service" -p ActiveState -p NRestarts >"$out/service/after-service.txt" || status=1
  sha256sum "$manifest" >"$out/service/active-manifest-after.sha256" || status=1
  if systemctl is-active --quiet "$service"; then
    python3 - "$root/tools/lightweight_promotion.py" >"$out/service/restore-response.json" 2>"$out/service/restore-response.stderr" <<'PY' || status=1
import importlib.util, json, sys
from pathlib import Path
s=importlib.util.spec_from_file_location('p',sys.argv[1]); m=importlib.util.module_from_spec(s); sys.modules['p']=m; s.loader.exec_module(m)
t=m.read_token(Path('/etc/ullm/openai-api-key'))
code,response,err=m._http_json('http://172.20.0.1:8000/v1/chat/completions',token=t,payload={'model':'ullm-qwen3.5-9b-aq4','messages':[{'role':'user','content':'Reply only: restored'}],'max_completion_tokens':8},timeout_seconds=45,gateway_container='open-webui')
if code != 200 or response is None or err: raise SystemExit(f'restore response failed {code} {err}')
print(json.dumps({'http_status':code,'content':m._extract_completion(response)},ensure_ascii=False))
PY
  else
    status=1
  fi
  metric >"$out/telemetry/after-restore.json" 2>&1 || true
  exit "$status"
}
trap restore EXIT

# Required pre-stop evidence.  active.json is recorded, never modified.
state before-stop
sha256sum "$manifest" >"$out/service/active-manifest-before.sha256"
grep -q '^a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd  ' "$out/service/active-manifest-before.sha256" || {
  echo 'active manifest SHA differs from authorized AQ4_0 value' >&2; exit 75;
}
metric >"$out/telemetry/before-stop.json"
systemctl is-active --quiet "$service" || { echo 'production service is not active' >&2; exit 75; }
systemctl show llama-qwen35-udq4.service -p ActiveState -p UnitFileState >"$out/preflight/llama-service.txt"
grep -qx 'ActiveState=inactive' "$out/preflight/llama-service.txt" && grep -qx 'UnitFileState=disabled' "$out/preflight/llama-service.txt" || {
  echo 'llama-qwen35-udq4.service must remain inactive and disabled' >&2; exit 75;
}

sudo_systemctl stop "$service" >"$out/service/stop.stdout" 2>"$out/service/stop.stderr"; stopped=1
for _ in $(seq 1 30); do ! systemctl is-active --quiet "$service" && ! fuser "$lock" >/dev/null 2>&1 && break; sleep 1; done
! systemctl is-active --quiet "$service" && ! fuser "$lock" >/dev/null 2>&1 || { echo 'service or lock remained busy after stop' >&2; exit 75; }
exec 9<>"$lock"; flock -n 9 || { echo 'R9700 lock raced busy' >&2; exit 75; }; held=1
metric >"$out/telemetry/after-lock.json"
for _ in $(seq 1 120); do
  metric >"$out/telemetry/thermal-gate-latest.json" 2>&1 || true
  edge=$(jq -r '.gpu_data[0].temperature.edge.value // empty' "$out/telemetry/thermal-gate-latest.json" 2>/dev/null || true)
  if [[ $edge =~ ^[0-9]+([.][0-9]+)?$ ]] && awk "BEGIN { exit !($edge <= 45) }"; then
    cp "$out/telemetry/thermal-gate-latest.json" "$out/telemetry/thermal-gate-pass.json"
    break
  fi
  sleep 2
done
[[ -f $out/telemetry/thermal-gate-pass.json ]] || { echo 'thermal gate did not reach edge <=45 C' >&2; exit 75; }

# The in-window runner contains its own sustained DPM warmup, continuous
# amd-smi samples, and a fail-closed clock-settled gate before timed groups.
started=$(date --iso-8601=ns)
HIP_VISIBLE_DEVICES=1 ULLM_HIP_VISIBLE_DEVICES=1 \
  HW_MB_MEMORY_PEAK_GBPS=640 HW_MB_BF16_PEAK_TFLOPS=191 HW_MB_FP8_PEAK_TFLOPS=383 \
  "$root/tools/run-hw-microbench-rdna4-cdna3.sh" --arch gfx1201 --amd-smi-gpu 2 --results-dir "$out/measurement"
finished=$(date --iso-8601=ns)
printf 'started_at=%s\nfinished_at=%s\n' "$started" "$finished" >"$out/window-wall-clock.txt"
